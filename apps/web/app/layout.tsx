import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MusicAI — Hybrid Guitar Transcription",
  description: "AI + vision + music theory guitar tab transcription",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru">
      <body>{children}</body>
    </html>
  );
}
