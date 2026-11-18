import { NavLink } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { LogOut } from "lucide-react";
function Topbar() {
  const baseClass =
    "px-4 py-2 rounded-xl transition text-sm font-medium";

  const activeClass = "bg-blue-600";
  const inactiveClass =
    "text-gray-300 hover:bg-gray-700 hover:text-white";

  const { logout } = useAuth()

  return (
    <div className="fixed top-0 left-0 right-0 h-16 
                    bg-gray-800/80 backdrop-blur-md 
                    border-b border-gray-700 
                    flex items-center px-6 z-50">

      <div className="justify-between flex w-full gap-4">
        <div className="px-3 py-2">

          <NavLink
            to="/"
            className={({ isActive }) =>
              `${baseClass} ${isActive ? activeClass : inactiveClass
              }`
            }
          >
            Chat
          </NavLink>

          <NavLink
            to="/documents"
            className={({ isActive }) =>
              `${baseClass} ${isActive ? activeClass : inactiveClass
              }`
            }
          >
            Documentos
          </NavLink>

        </div>
        
        <button
          onClick={logout}
          className="px-3 py-2 bg-gray-700 hover:bg-gray-800 rounded-xl text-sm"
        >
          <LogOut className="cursor-pointer hover:text-blue-500" size={18} />
        </button>

      </div>
    </div>
  );
}

export default Topbar;
