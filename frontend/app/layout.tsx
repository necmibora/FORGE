import "./globals.css";
import type { Metadata } from "next";
import { Providers } from "./providers";
import { Nav } from "@/components/Nav";

export const metadata: Metadata = {
  title: "FORGE",
  description: "Function-calling Open-source Runtime for Grounded Evaluation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="tr">
      <body className="min-h-screen">
        <Providers>
          <Nav />
          <main className="max-w-6xl mx-auto px-6 py-6">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
