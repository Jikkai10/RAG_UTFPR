import { createContext, useContext, useState, useEffect } from "react"
import { apiFetch } from "../Api"

const AuthContext = createContext()

export function AuthProvider({ children }) {

  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    checkUser()
  }, [])

  async function checkUser() {

    const token = localStorage.getItem("token")

    if (!token) {
      setLoading(false)
      return
    }

    try {
      const res = await apiFetch("/me")

      if (!res.ok) throw new Error()

      const data = await res.json()

      setUser(data)

    } catch {
      logout()
    }

    setLoading(false)
  }

  async function login(email, password) {

    const res = await apiFetch("/login", {
      method: "POST",
      body: JSON.stringify({ email, password })
    })

    if (!res.ok) {
      throw new Error("Login inválido")
    }

    const data = await res.json()

    localStorage.setItem("token", data.access_token)

    setUser(data.user)
  }

  function logout() {
    localStorage.removeItem("token")
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}