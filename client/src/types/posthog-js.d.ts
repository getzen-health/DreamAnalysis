declare module "posthog-js" {
  interface PostHogConfig {
    api_host?: string;
    autocapture?: boolean;
    capture_pageview?: boolean;
    persistence?: string;
    disable_session_recording?: boolean;
    loaded?: (ph: PostHog) => void;
    [key: string]: unknown;
  }

  interface PostHog {
    capture: (event: string, properties?: Record<string, unknown>) => void;
    identify: (distinctId: string, properties?: Record<string, unknown>) => void;
    reset: () => void;
    init: (apiKey: string, config?: PostHogConfig) => void;
  }

  const posthog: PostHog;
  export default posthog;
}
