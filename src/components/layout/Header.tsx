import { Brain, Sparkles } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Home", href: "/" },
  { name: "Query", href: "/query" },
  { name: "Dashboard", href: "/dashboard" },
  { name: "Review Analysis", href: "/review-analysis" },
];

export function Header() {
  const location = useLocation();

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center">
        <Link to="/" className="flex items-center space-x-2">
          <div className="relative">
            <Brain className="h-8 w-8 text-primary" />
            <Sparkles className="absolute -top-1 -right-1 h-4 w-4 text-primary-glow" />
          </div>
          <span className="hidden font-bold sm:inline-block bg-gradient-primary bg-clip-text text-transparent">
            AI Recommend
          </span>
        </Link>
        
        <nav className="flex items-center space-x-6 text-sm font-medium ml-8">
          {navigation.map((item) => (
            <Link
              key={item.href}
              to={item.href}
              className={cn(
                "transition-colors hover:text-primary",
                location.pathname === item.href
                  ? "text-primary"
                  : "text-muted-foreground"
              )}
            >
              {item.name}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}