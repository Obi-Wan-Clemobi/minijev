import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/Nav";
import { StoreProvider } from "@/lib/store";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "minijev playground",
  description: "Typed, calibrated decisions from one forward pass of a small open model, on your laptop.",
};

// Dark by default; the saved choice is applied before paint to avoid a flash.
const themeScript = `try{if(localStorage.getItem('mj-theme')==='light')document.documentElement.classList.remove('dark')}catch(e){}`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`dark ${geistSans.variable} ${geistMono.variable} h-full antialiased`} suppressHydrationWarning>
      <head><script dangerouslySetInnerHTML={{ __html: themeScript }} /></head>
      <body className="min-h-full flex flex-col bg-bg text-fg">
        <StoreProvider>
          <Nav />
          <main className="flex-1 flex flex-col">{children}</main>
        </StoreProvider>
      </body>
    </html>
  );
}
