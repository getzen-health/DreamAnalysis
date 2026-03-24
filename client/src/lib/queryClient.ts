import { QueryClient, QueryCache, QueryFunction } from "@tanstack/react-query";
import { Capacitor } from "@capacitor/core";
import { toast } from "@/hooks/use-toast";

const isNative = Capacitor.isNativePlatform();

/**
 * On native Capacitor apps, relative URLs like /api/auth/register resolve to
 * file:// which fails. Prefix with the production backend URL.
 */
export function resolveUrl(url: string): string {
  if (isNative && url.startsWith("/")) {
    const base = import.meta.env.VITE_EXPRESS_URL || "https://dream-analysis.vercel.app";
    return `${base}${url}`;
  }
  return url;
}

/**
 * On native, credentials: "include" causes CORS failure because the server
 * returns Access-Control-Allow-Origin: * which is incompatible with credentials.
 * Native uses JWT tokens via Authorization header instead of cookies.
 */
const fetchCredentials: RequestCredentials = isNative ? "omit" : "include";

async function throwIfResNotOk(res: Response) {
  if (!res.ok) {
    let message: string;
    try {
      const body = await res.json();
      message = body.error || body.message || res.statusText;
    } catch {
      message = (await res.text().catch(() => '')) || res.statusText;
    }
    const err = new Error(message) as Error & { status: number };
    err.status = res.status;
    throw err;
  }
}

/** Read JWT token from localStorage (set after login/register). */
function getStoredToken(): string | null {
  try { return localStorage.getItem("auth_token"); } catch { return null; }
}

export async function apiRequest(
  method: string,
  url: string,
  data?: unknown | undefined,
): Promise<Response> {
  const headers: Record<string, string> = {};
  const token = getStoredToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  let body: string | undefined;
  if (data) {
    body = JSON.stringify(data);
    headers["Content-Type"] = "application/json";
    // Vercel's serverless runtime has a bug where req.body is consumed but
    // inaccessible. Send body as base64 header as a workaround.
    // Skip for large payloads (>6KB) — HTTP headers have ~8KB size limits.
    // Large payloads (e.g. image uploads) fall through to the stream parser.
    if (body.length < 6000) {
      headers["x-body-b64"] = btoa(body);
    }
  }

  const res = await fetch(resolveUrl(url), {
    method,
    headers,
    body,
    credentials: fetchCredentials,
  });

  await throwIfResNotOk(res);
  return res;
}

type UnauthorizedBehavior = "returnNull" | "throw";
export const getQueryFn: <T>(options: {
  on401: UnauthorizedBehavior;
}) => QueryFunction<T> =
  ({ on401: unauthorizedBehavior }) =>
  async ({ queryKey }) => {
    const headers: Record<string, string> = {};
    const token = getStoredToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const res = await fetch(resolveUrl(queryKey.join("/") as string), {
      credentials: fetchCredentials,
      headers,
    });

    if (unauthorizedBehavior === "returnNull" && res.status === 401) {
      return null;
    }

    await throwIfResNotOk(res);
    return await res.json();
  };

// Throttle error toasts so rapid failures don't spam the user
let lastErrorToastMs = 0;

export const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError: (error, query) => {
      const now = Date.now();
      if (now - lastErrorToastMs < 5000) return; // 5s cooldown

      const status = (error as any)?.status as number | undefined;
      if (status === 401) return; // auth errors handled by ProtectedRoute
      // 404 = no data yet (expected for new users) — don't alarm them
      if (status === 404) return;
      // Suppress ALL background query error toasts — these are not user-initiated
      // actions. Mutation errors are handled by each mutation's own catch block.
      return;
    },
  }),
  defaultOptions: {
    queries: {
      queryFn: getQueryFn({ on401: "throw" }),
      refetchInterval: false,
      refetchOnWindowFocus: true,
      staleTime: 60_000,
      retry: false,
    },
    mutations: {
      retry: false,
    },
  },
});
