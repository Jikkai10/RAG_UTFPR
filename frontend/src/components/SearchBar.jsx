import { useState, useRef, useEffect } from "react";
import { Search } from "lucide-react";

export default function SearchBar({ setValue }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const inputRef = useRef(null);

  // fechar ao clicar fora
  useEffect(() => {
    function handleClickOutside(e) {
      if (ref.current && !ref.current.contains(e.target)) {
        setOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // focar quando abrir
  useEffect(() => {
    if (open) {
      inputRef.current?.focus();
    }
  }, [open]);

  return (
    <div ref={ref} className="w-full flex items-center">

      <div
        className={`
        flex items-center bg-gray-800 border border-gray-700 rounded-full
        transition-all duration-300 ease-in-out
        ${open ? "w-full px-2" : "w-10 justify-center"}
        `}
      >

        <button
          className="p-2 text-gray-400 shrink-0"
          onClick={() => setOpen((v) => !v)}
        >
          <Search size={20} />
        </button>

        <input
          ref={inputRef}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") setOpen(false);
          }}
          placeholder="Buscar..."
          className={`
            bg-transparent outline-none text-white w-full 
            transition-all duration-200
            ${open ? "opacity-100" : "opacity-0 w-0 ml-0"}
          `}
        />

      </div>

    </div>
  );
}