import {
  Plus,
  ChevronLeft,
  ChevronRight,
  Settings2,
  MoreHorizontal,
  Trash2,
  Pencil,
  MessageCircle,
  MessagesSquare,
} from "lucide-react";
import { useState, useRef, useEffect } from "react";

function Sidebar({ chats, activeChat, setActiveChat, deleteChat, updateChat }) {

  const [open, setOpen] = useState(true);
  const [menuOpen, setMenuOpen] = useState(null);
  const [editingChat, setEditingChat] = useState(null);
  const [title, setTitle] = useState("");

  const menuRef = useRef(null);

  // fechar menu ao clicar fora
  useEffect(() => {
    function handleClick(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(null);
      }
    }

    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const removeChat = (id) => {
    if (id === activeChat) setActiveChat(null);
    deleteChat(id);
  };

  const startRename = (chat) => {
    setEditingChat(chat.id);
    setTitle(chat.title);
    setMenuOpen(null);
  };

  const saveRename = () => {
    updateChat(editingChat, title);
    setEditingChat(null);
  };

  return (
    <div
      className={`relative flex flex-col bg-gray-950 border-r border-gray-800
      transition-all duration-300
      ${open ? "w-64" : "w-12"}`}
    >


      <button
        onClick={() => setOpen(!open)}
        className="absolute -right-3 top-11 z-10
        bg-gray-900 border border-gray-700
        rounded-full p-1 hover:bg-gray-600 shadow-md"
      >
        {open ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
      </button>


      <button
        title={open ? "" : "Novo chat"}
        onClick={() => setActiveChat(null)}
        className={`mt-2 ${open ? "mx-2" : "mx-1 justify-center"} p-2 rounded-lg hover:bg-gray-800 flex items-center gap-2`}
      >
        <Plus size={18} />
        {open && <span>Novo chat</span>}
      </button>

      {open ? (
        <div className="text-xs px-3 pt-2 pb-1 text-gray-400 uppercase">
          Chats
        </div>
      ) :
        <div className=" px-3  h-8  flex items-center">
          <MessagesSquare className="text-gray-400" size={14} />
        </div>
      }

      <div className="flex-1 overflow-y-auto">

        {chats.map((chat, i) => (

          <div
            key={chat.id}
            title={chat.title}
            className={`group relative ${open ? "mx-2" : "mx-1"} mb-1 rounded-lg
            ${activeChat === chat.id
                ? "bg-gray-700"
                : "hover:bg-gray-800"}`}
          >

            <div
              onClick={() => setActiveChat(chat.id)}
              className={"flex items-center justify-between p-2 text-sm cursor-pointer"}
            >


              {editingChat === chat.id ? (

                <input
                  autoFocus
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  onBlur={saveRename}
                  onKeyDown={(e) => e.key === "Enter" && saveRename()}
                  className={`bg-gray-800 text-sm  py-1 rounded w-full outline-none`}
                />

              ) : (

                <span className={`truncate w-full ${open ? "" : "justify-center"}`}>
                  {open ? chat.title :
                    <div className="relative flex justify-center">
                      <MessageCircle className="text-gray-300"  size={18} />
                      <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-gray-200 text-[9px] text-center">
                        {i+1}
                      </span>
                    </div>}
                </span>

              )}

              {open && editingChat !== chat.id && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setMenuOpen(menuOpen === chat.id ? null : chat.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-gray-700"
                >
                  <MoreHorizontal size={16} />
                </button>
              )}

            </div>

            {/* menu */}
            {menuOpen === chat.id && (
              <div
                ref={menuRef}
                className="absolute right-2 top-8 bg-gray-900 border border-gray-700
                rounded-md shadow-lg text-sm z-10 w-36"
              >

                <button
                  onClick={() => startRename(chat)}
                  className="flex items-center gap-2 w-full px-3 py-2 hover:bg-gray-800"
                >
                  <Pencil size={14} />
                  Renomear
                </button>

                <button
                  onClick={() => removeChat(chat.id)}
                  className="flex items-center gap-2 w-full px-3 py-2 hover:bg-gray-800 text-red-400"
                >
                  <Trash2 size={14} />
                  Remover
                </button>

              </div>
            )}

          </div>

        ))}

      </div>

    </div>
  );
}

export default Sidebar;