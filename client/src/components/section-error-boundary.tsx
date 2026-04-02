import { Component, type ReactNode } from "react";

interface Props {
  label: string;
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

/**
 * Lightweight section-level error boundary.
 * Wraps individual cards/sections so one failing component
 * doesn't crash the entire page.
 */
export class SectionErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error) {
    console.error(`[SectionErrorBoundary] "${this.props.label}" crashed:`, error.message);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="rounded-xl border border-destructive/20 bg-destructive/5 p-4 text-center text-sm text-muted-foreground">
          <p className="font-medium text-destructive/80">Couldn't load {this.props.label}</p>
          <button
            className="mt-2 text-xs underline underline-offset-2 hover:text-foreground"
            onClick={() => this.setState({ hasError: false })}
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
