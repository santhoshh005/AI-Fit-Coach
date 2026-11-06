import { Toaster } from "@/components/ui/toaster";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ThemeProvider } from "next-themes";
import { AuthProvider, useAuth } from "./contexts/AuthContext";
import { Layout } from "./components/Layout";
import { lazy, Suspense } from "react";
import Index from "./pages/Index";

// Route-level code splitting: each page (and heavy deps like MediaPipe and
// Recharts) downloads only when the route is first visited.
const Dashboard = lazy(() => import("./pages/Dashboard"));
const AITrainer = lazy(() => import("./pages/AITrainer"));
const ExerciseGuide = lazy(() => import("./pages/ExerciseGuide"));
const WorkoutTracker = lazy(() => import("./pages/WorkoutTracker"));
const NutritionTracker = lazy(() => import("./pages/NutritionTracker"));
const HealthMonitoring = lazy(() => import("./pages/HealthMonitoring"));
const Login = lazy(() => import("./pages/Login"));
const Signup = lazy(() => import("./pages/Signup"));
const Profile = lazy(() => import("./pages/Profile"));
const NotFound = lazy(() => import("./pages/NotFound"));

const PageFallback = () => (
  <div role="status" className="min-h-[50vh] flex items-center justify-center text-muted-foreground">Loading...</div>
);

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="min-h-screen flex items-center justify-center text-muted-foreground">Loading...</div>;
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function PublicRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="min-h-screen flex items-center justify-center text-muted-foreground">Loading...</div>;
  if (user) return <Navigate to="/dashboard" replace />;
  return <>{children}</>;
}

const App = () => (
  <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false}>
    <Toaster />
    <BrowserRouter>
      <AuthProvider>
        <Layout>
          <Suspense fallback={<PageFallback />}>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
            <Route path="/signup" element={<PublicRoute><Signup /></PublicRoute>} />
            <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
            <Route path="/ai-trainer" element={<ProtectedRoute><AITrainer /></ProtectedRoute>} />
            <Route path="/exercises" element={<ExerciseGuide />} />
            <Route path="/workouts" element={<ProtectedRoute><WorkoutTracker /></ProtectedRoute>} />
            <Route path="/nutrition" element={<ProtectedRoute><NutritionTracker /></ProtectedRoute>} />
            <Route path="/health" element={<ProtectedRoute><HealthMonitoring /></ProtectedRoute>} />
            <Route path="/analytics" element={<Navigate to="/health" replace />} />
            <Route path="/profile" element={<ProtectedRoute><Profile /></ProtectedRoute>} />
            <Route path="*" element={<NotFound />} />
          </Routes>
          </Suspense>
        </Layout>
      </AuthProvider>
    </BrowserRouter>
  </ThemeProvider>
);

export default App;
