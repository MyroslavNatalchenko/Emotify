export function Footer() {
  return (
    <footer className="w-full border-t border-white/5 bg-black py-8 mt-auto">
      <div className="mx-auto flex max-w-md md:max-w-7xl flex-col items-center justify-between gap-4 px-6 md:flex-row text-xs text-neutral-600">
        <p>&copy; 2024 Emotify</p>

        <div className="flex gap-6">
          <span className="hover:text-green-500 cursor-pointer transition-colors p-2">Privacy</span>
          <span className="hover:text-green-500 cursor-pointer transition-colors p-2">Terms</span>
        </div>
      </div>
    </footer>
  );
}