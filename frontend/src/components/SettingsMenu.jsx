import { useState, useRef, useEffect } from "react";
import { Settings, MessageSquare, FileText, BarChart3, LogOut } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function SettingsMenu() {
  const [open, setOpen] = useState(false);
  const ref = useRef();
  const navigate = useNavigate();
  const { logout, user } = useAuth();

  // fecha o menu ao clicar fora
  useEffect(() => {
    function handleClick(e) {
      if (ref.current && !ref.current.contains(e.target)) {
        setOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  return (
    <div className="fixed top-4 right-4 z-50" ref={ref}>
      
      {/* botão settings */}
      <motion.button
        onClick={() => setOpen(!open)}
        whileTap={{ scale: 0.9 }}
        animate={{ rotate: open ? 90 : 0 }}
        transition={{ duration: 0.25 }}
        className="p-2 rounded-xl bg-gray-800 hover:bg-gray-700 transition shadow-md"
      >
        <Settings className="w-5 h-5 text-white" />
      </motion.button>

      {/* menu dropdown */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 6, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="absolute right-full top-4 mt-2 w-48 bg-gray-900 border border-gray-700 rounded-2xl shadow-xl overflow-hidden"
          >
            <MenuItem
              icon={<MessageSquare size={18} />}
              label="Chats"
              onClick={() => {
                navigate("/");
                setOpen(false);
              }}
            />

            <MenuItem
              icon={<FileText size={18} />}
              label="Documents"
              onClick={() => {
                navigate("/documents");
                setOpen(false);
              }}
            />

            {(user?.role === "admin") && (
              <MenuItem
                icon={<BarChart3 size={18} />}
                label="Avaliação"
                onClick={() => {
                  navigate("/evaluation");
                  setOpen(false);
                }}
              />
            )}

            <div className="border-t border-gray-700 my-1"></div>

            <MenuItem
              icon={<LogOut size={18} />}
              label="Sair"
              danger
              onClick={() => {
                logout();
                setOpen(false);
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function MenuItem({ icon, label, onClick, danger }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-3 w-full px-4 py-3 text-sm transition
      ${
        danger
          ? "text-red-400 hover:bg-red-500/10"
          : "text-gray-200 hover:bg-gray-800"
      }`}
    >
      {icon}
      {label}
    </button>
  );
}