/**
 * Shared test utilities — wraps components with the providers required
 * to render pages in isolation (QueryClient + Theme).
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, type RenderOptions } from "@testing-library/react";

function makeClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
}

/** Drop-in replacement for @testing-library/react `render`.
 *  Wraps the component in QueryClientProvider so hooks like
 *  useQuery / useQueryClient work without a real server. */
export function renderWithProviders(
  ui: React.ReactElement,
  options?: RenderOptions
) {
  const client = makeClient();
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
    options
  );
}
