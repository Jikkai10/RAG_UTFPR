import {
  createBrowserRouter,
  RouterProvider,
  Outlet,
} from "react-router-dom";
import ProtectedRoute from "./components/ProtectedRoute";
import ChatBot from "./pages/ChatBot";
import Documents from "./pages/Documents";
import Evaluation from "./pages/Evaluation";
import Login from "./pages/Login";
import Topbar from "./components/Topbar";
import SettingsMenu from "./components/SettingsMenu";
function Layout() {
  return (
    <div className="h-screen bg-gray-900 text-white flex flex-col">
      <SettingsMenu />


      <Outlet />

    </div>
  );
}

const router = createBrowserRouter([
  {
    element: (
      <ProtectedRoute>
        <Layout />
      </ProtectedRoute>
    ),
    children: [
      { path: "/", element: <ChatBot /> },
      { path: "/documents", element: <Documents /> },
      { path: "/evaluation", element: <Evaluation /> },
    ],
  },
  {
    path: "/login",
    element: <Login />,
  }
]);

function App() {
  return (

    <RouterProvider router={router} />

  );
}

export default App;
