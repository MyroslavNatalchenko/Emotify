'use client';

import Link from 'next/link';
import { Menu, Heart } from 'lucide-react';
import { useState } from 'react';

export function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <nav className="fixed top-0 right-0 left-0 z-50 border-b border-slate-200 bg-white/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-2 transition-opacity hover:opacity-80">
          <div className="flex h-9 w-9 items-center justify-center rounded-2xl bg-emerald-500 text-white shadow-lg shadow-emerald-200">
            <Heart className="h-5 w-5" fill="currentColor" />
          </div>
          <div className="flex flex-col">
            <span className="text-lg leading-none font-bold tracking-tight text-slate-800">
              Emotify
            </span>
          </div>
        </Link>

        {/* Mobile Menu Toggle */}
        <button
          className="p-2 text-slate-600 transition-colors hover:text-emerald-500 md:hidden"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
          <Menu className="h-7 w-7" />
        </button>

        {/* Desktop Links */}
        <div className="hidden gap-8 md:flex">
          <Link href="/" className="text-sm font-semibold text-slate-600 hover:text-emerald-500">
            Dashboard
          </Link>
          <Link
            href="#"
            className="cursor-not-allowed text-sm font-semibold text-slate-600 opacity-50 hover:text-emerald-500"
          >
            Historia
          </Link>
        </div>
      </div>

      {/* Mobile Dropdown */}
      {isMenuOpen && (
        <div className="animate-in slide-in-from-top-2 absolute top-16 left-0 flex w-full flex-col gap-4 border-b border-slate-200 bg-white p-6 shadow-xl md:hidden">
          <Link
            href="/"
            onClick={() => setIsMenuOpen(false)}
            className="text-base font-bold text-slate-800"
          >
            Strona Główna
          </Link>
          <div className="h-px bg-slate-100" />
          <p className="text-center text-[10px] font-medium text-slate-400">Emotify</p>
        </div>
      )}
    </nav>
  );
}
