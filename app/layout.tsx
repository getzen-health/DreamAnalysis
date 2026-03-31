import type { Metadata } from "next";
import { Inter, Orbitron } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-inter",
  display: "swap",
});

const orbitron = Orbitron({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800", "900"],
  variable: "--font-orbitron",
  display: "swap",
});

export const metadata: Metadata = {
  title: "getZen Health — Know Your Mind",
  description: "Voice, dreams, nutrition, sleep, brain, and emotions — all unified. getZen Health reads your inner world every day.",
  openGraph: {
    title: "getZen Health — Know Your Mind",
    description: "The only wellness app that connects your brain, voice, dreams, nutrition, and emotions into one daily picture.",
    url: "https://getzen.health",
    siteName: "getZen Health",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${orbitron.variable}`}>
      <body>{children}</body>
    </html>
  );
}
