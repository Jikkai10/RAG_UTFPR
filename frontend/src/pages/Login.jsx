import { useAuth } from "../context/AuthContext"
import { useState } from "react"
import { useNavigate } from "react-router-dom"

export default function Login() {
    const { login } = useAuth()
    const navigate = useNavigate()
    const [email, setEmail] = useState("")
    const [password, setPassword] = useState("")
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState("")

    async function handleLogin(e) {
        e.preventDefault()
        setError("")
        setLoading(true)

        try {
            await login(email, password)
            navigate("/", { replace: true })
        } catch (err) {
            setError("Email ou senha inválidos")
        }

        setLoading(false)
    }

    return (
        <div className="h-screen flex items-center justify-center bg-gray-900 text-white">

            <div className="w-full max-w-md bg-gray-800 p-8 rounded-2xl shadow-lg border border-gray-700">

                <h1 className="text-3xl font-bold mb-6 text-center">
                    Login
                </h1>

                <form onSubmit={handleLogin} className="space-y-4">

                    <div>
                        <label className="block text-sm mb-1">
                            Email
                        </label>

                        <input
                            type="email"
                            placeholder="seu@email.com"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="w-full px-4 py-2 rounded-lg bg-gray-900 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                    </div>

                    <div>
                        <label className="block text-sm mb-1">
                            Senha
                        </label>

                        <input
                            type="password"
                            placeholder="********"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="w-full px-4 py-2 rounded-lg bg-gray-900 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                    </div>

                    {error && (
                        <div className="text-red-400 text-sm">
                            {error}
                        </div>
                    )}

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-blue-600 hover:bg-blue-700 transition py-2 rounded-lg font-medium"
                    >
                        {loading ? "Entrando..." : "Entrar"}
                    </button>

                </form>
            </div>

        </div>
    )
}