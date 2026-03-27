import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: ["./client/index.html", "./client/src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        card: {
          DEFAULT: "var(--card)",
          foreground: "var(--card-foreground)",
        },
        popover: {
          DEFAULT: "var(--popover)",
          foreground: "var(--popover-foreground)",
        },
        primary: {
          DEFAULT: "var(--primary)",
          foreground: "var(--primary-foreground)",
        },
        secondary: {
          DEFAULT: "var(--secondary)",
          foreground: "var(--secondary-foreground)",
        },
        muted: {
          DEFAULT: "var(--muted)",
          foreground: "var(--muted-foreground)",
        },
        accent: {
          DEFAULT: "var(--accent)",
          foreground: "var(--accent-foreground)",
        },
        destructive: {
          DEFAULT: "var(--destructive)",
          foreground: "var(--destructive-foreground)",
        },
        success: {
          DEFAULT: "var(--success)",
          foreground: "var(--success-foreground)",
        },
        warning: {
          DEFAULT: "var(--warning)",
          foreground: "var(--warning-foreground)",
        },
        border: "var(--border)",
        input: "var(--input)",
        ring: "var(--ring)",
        chart: {
          "1": "var(--chart-1)",
          "2": "var(--chart-2)",
          "3": "var(--chart-3)",
          "4": "var(--chart-4)",
          "5": "var(--chart-5)",
        },
        sidebar: {
          DEFAULT: "var(--sidebar-background)",
          foreground: "var(--sidebar-foreground)",
          primary: "var(--sidebar-primary)",
          "primary-foreground": "var(--sidebar-primary-foreground)",
          accent: "var(--sidebar-accent)",
          "accent-foreground": "var(--sidebar-accent-foreground)",
          border: "var(--sidebar-border)",
          ring: "var(--sidebar-ring)",
        },
        neural: {
          blue: "var(--neural-blue)",
          cyan: "var(--neural-cyan)",
          purple: "var(--neural-purple)",
          green: "var(--neural-green)",
        },
        // Bevel-style score colors
        'ndw-bg': '#FAFAFA',
        'ndw-card': '#FFFFFF',
        'ndw-border': '#E5E7EB',
        'ndw-text': '#131313',
        'ndw-muted': '#6B7280',
        'ndw-recovery': '#34D399',
        'ndw-sleep': '#60A5FA',
        'ndw-strain': '#FB923C',
        'ndw-stress': '#F87171',
        'ndw-nutrition': '#F59E0B',
        'ndw-energy': '#34D399',
      },
      fontFamily: {
        sans: ["'Public Sans'", "system-ui", "-apple-system", "sans-serif"],
        serif: ["Georgia", "serif"],
        mono: ["var(--font-mono)"],
        futuristic: ["'Public Sans'", "system-ui", "-apple-system", "sans-serif"],
      },
      keyframes: {
        "accordion-down": {
          from: {
            height: "0",
          },
          to: {
            height: "var(--radix-accordion-content-height)",
          },
        },
        "accordion-up": {
          from: {
            height: "var(--radix-accordion-content-height)",
          },
          to: {
            height: "0",
          },
        },
        'pulse-glow': {
          '0%, 100%': {
            boxShadow: '0 0 20px rgba(52, 211, 153, 0.2)',
            transform: 'scale(1)'
          },
          '50%': {
            boxShadow: '0 0 30px rgba(52, 211, 153, 0.4)',
            transform: 'scale(1.02)'
          },
        },
        'neural-flow': {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100vw)' },
        },
        'brain-wave': {
          '0%, 100%': { transform: 'scaleY(1)' },
          '25%': { transform: 'scaleY(1.5)' },
          '50%': { transform: 'scaleY(0.8)' },
          '75%': { transform: 'scaleY(1.2)' },
        },
        'data-stream': {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '50%': { opacity: '1' },
          '100%': { opacity: '0', transform: 'translateY(-20px)' },
        },
        'float': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'neural-flow': 'neural-flow 3s linear infinite',
        'brain-wave': 'brain-wave 1.5s ease-in-out infinite',
        'data-stream': 'data-stream 2s linear infinite',
        'float': 'float 3s ease-in-out infinite',
      },
    },
  },
  plugins: [require("tailwindcss-animate"), require("@tailwindcss/typography")],
} satisfies Config;
