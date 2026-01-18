'use client';

import Link from "next/link";
import { Headphones, Github, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { usePathname } from "next/navigation";
import { useState } from "react";

export function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/5 bg-black/60 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-md md:max-w-7xl items-center justify-between px-6">

        <Link href="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
          <div className="flex h-9 w-9 items-center justify-center rounded-2xl bg-green-500/20 text-green-500">
            <Headphones className="h-5 w-5" />
          </div>
          <span className="text-lg font-bold text-white tracking-tight">Emotify</span>
        </Link>

        <button
            className="md:hidden text-neutral-300 hover:text-white"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
            <Menu className="h-7 w-7" />
        </button>

        <div className="hidden md:flex items-center gap-4">
          <Link href="https://github.com" target="_blank">
            <Button variant="ghost" size="icon" className="text-neutral-400 hover:text-white hover:bg-white/10 rounded-xl">
              <Github className="h-5 w-5" />
            </Button>
          </Link>
        </div>
      </div>

      {isMenuOpen && (
        <div className="absolute top-16 left-0 w-full bg-neutral-900 border-b border-white/10 p-4 flex flex-col gap-4 md:hidden shadow-2xl animate-in slide-in-from-top-5">
           <Link href="/" onClick={() => setIsMenuOpen(false)} className="text-sm font-medium text-neutral-300 hover:text-green-500">
             Strona Główna
           </Link>
           <Link href="/dashboard" onClick={() => setIsMenuOpen(false)} className="text-sm font-medium text-neutral-300 hover:text-green-500">
             Dashboard
           </Link>
           <div className="h-px bg-white/10 my-1" />
           <p className="text-xs text-neutral-500 text-center">Emotify v1.0</p>
        </div>
      )}
    </nav>
  );
}