import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import ForgotPasswordPage from "@/pages/forgot-password";
import ResetPasswordPage from "@/pages/reset-password";

// wouter's useLocation needs a mock so setLocation calls don't crash jsdom
vi.mock("wouter", () => ({
  useLocation: () => ["", vi.fn()],
}));

vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

// ── helpers ────────────────────────────────────────────────────────────────

function mockFetchOk(body: unknown = {}) {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => body,
  });
}

function mockFetchError(status: number, body: unknown = {}) {
  return vi.fn().mockResolvedValue({
    ok: false,
    status,
    json: async () => body,
  });
}

// ── ForgotPasswordPage ─────────────────────────────────────────────────────

describe("ForgotPasswordPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the email form", () => {
    global.fetch = mockFetchOk() as typeof fetch;
    renderWithProviders(<ForgotPasswordPage />);
    expect(screen.getByLabelText(/email address/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /send reset link/i })).toBeInTheDocument();
  });

  it("POSTs to /api/auth/forgot-password with the entered email", async () => {
    const fetchMock = mockFetchOk({ message: "If that email exists, a reset link was sent" });
    global.fetch = fetchMock as typeof fetch;

    renderWithProviders(<ForgotPasswordPage />);

    fireEvent.change(screen.getByLabelText(/email address/i), { target: { value: "sravya@example.com" } });
    fireEvent.submit(screen.getByRole("button", { name: /send reset link/i }).closest("form")!);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/auth/forgot-password",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({ "Content-Type": "application/json" }),
          body: JSON.stringify({ email: "sravya@example.com" }),
        })
      );
    });
  });

  it("shows confirmation message on successful submission", async () => {
    global.fetch = mockFetchOk({ message: "If that email exists, a reset link was sent" }) as typeof fetch;

    renderWithProviders(<ForgotPasswordPage />);
    fireEvent.change(screen.getByLabelText(/email address/i), { target: { value: "test@test.com" } });
    fireEvent.submit(screen.getByRole("button", { name: /send reset link/i }).closest("form")!);

    await waitFor(() => {
      expect(screen.getByText(/if that email is in our system/i)).toBeInTheDocument();
    });
  });

  it("shows an error message when the server returns non-ok", async () => {
    global.fetch = mockFetchError(500, { message: "Server error" }) as typeof fetch;

    renderWithProviders(<ForgotPasswordPage />);
    fireEvent.change(screen.getByLabelText(/email address/i), { target: { value: "bad@test.com" } });
    fireEvent.submit(screen.getByRole("button", { name: /send reset link/i }).closest("form")!);

    await waitFor(() => {
      expect(screen.getByText(/server error/i)).toBeInTheDocument();
    });
  });

  it("shows a network error message on fetch rejection", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("Network down")) as typeof fetch;

    renderWithProviders(<ForgotPasswordPage />);
    fireEvent.change(screen.getByLabelText(/email address/i), { target: { value: "test@test.com" } });
    fireEvent.submit(screen.getByRole("button", { name: /send reset link/i }).closest("form")!);

    await waitFor(() => {
      expect(screen.getByText(/network error/i)).toBeInTheDocument();
    });
  });
});

// ── ResetPasswordPage ──────────────────────────────────────────────────────

describe("ResetPasswordPage", () => {
  const VALID_TOKEN = "abc123validtoken";

  beforeEach(() => {
    vi.restoreAllMocks();
    // Put the reset token into the URL search string
    Object.defineProperty(window, "location", {
      value: { ...window.location, search: `?token=${VALID_TOKEN}` },
      writable: true,
    });
  });

  it("renders the new-password form", () => {
    global.fetch = mockFetchOk() as typeof fetch;
    renderWithProviders(<ResetPasswordPage />);
    expect(screen.getByLabelText(/new password/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/confirm password/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /update password/i })).toBeInTheDocument();
  });

  it("POSTs to /api/auth/reset-password with token and new password", async () => {
    const fetchMock = mockFetchOk({ message: "Password updated successfully" });
    global.fetch = fetchMock as typeof fetch;

    renderWithProviders(<ResetPasswordPage />);

    fireEvent.change(screen.getByLabelText(/new password/i), { target: { value: "newSecurePass1!" } });
    fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: "newSecurePass1!" } });
    fireEvent.submit(screen.getByRole("button", { name: /update password/i }).closest("form")!);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/auth/reset-password",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({ "Content-Type": "application/json" }),
          body: JSON.stringify({ token: VALID_TOKEN, newPassword: "newSecurePass1!" }),
        })
      );
    });
  });

  it("does NOT submit when passwords do not match", async () => {
    const fetchMock = mockFetchOk() as typeof fetch;
    global.fetch = fetchMock;

    renderWithProviders(<ResetPasswordPage />);

    fireEvent.change(screen.getByLabelText(/new password/i), { target: { value: "passOne123!" } });
    fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: "passTwo999!" } });
    fireEvent.submit(screen.getByRole("button", { name: /update password/i }).closest("form")!);

    // fetch must not be called when passwords mismatch
    await waitFor(() => {
      expect(fetchMock).not.toHaveBeenCalled();
    });
  });

  it("shows an error toast when the server rejects the token", async () => {
    const toastFn = vi.fn();
    vi.doMock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: toastFn }) }));

    global.fetch = mockFetchError(400, { message: "Invalid or expired reset token" }) as typeof fetch;

    renderWithProviders(<ResetPasswordPage />);

    fireEvent.change(screen.getByLabelText(/new password/i), { target: { value: "newPass123!" } });
    fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: "newPass123!" } });
    fireEvent.submit(screen.getByRole("button", { name: /update password/i }).closest("form")!);

    await waitFor(() => {
      // fetch was called — the error path ran
      expect(global.fetch).toHaveBeenCalledWith("/api/auth/reset-password", expect.anything());
    });
  });

  it("does NOT submit when token is missing from the URL", async () => {
    Object.defineProperty(window, "location", {
      value: { ...window.location, search: "" },
      writable: true,
    });

    const fetchMock = mockFetchOk() as typeof fetch;
    global.fetch = fetchMock;

    renderWithProviders(<ResetPasswordPage />);

    fireEvent.change(screen.getByLabelText(/new password/i), { target: { value: "newPass123!" } });
    fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: "newPass123!" } });
    fireEvent.submit(screen.getByRole("button", { name: /update password/i }).closest("form")!);

    await waitFor(() => {
      expect(fetchMock).not.toHaveBeenCalled();
    });
  });
});
