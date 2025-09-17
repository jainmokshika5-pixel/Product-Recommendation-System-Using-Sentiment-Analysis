import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Brain, Sparkles, Target, TrendingUp, Zap } from "lucide-react";
import { Link } from "react-router-dom";

const features = [
  {
    icon: Brain,
    title: "Smart NLP Processing",
    description: "Advanced natural language processing to understand your product queries",
  },
  {
    icon: Target,
    title: "Sentiment Analysis", 
    description: "CNN+BiLSTM deep learning model for accurate sentiment classification",
  },
  {
    icon: TrendingUp,
    title: "Intelligent Ranking",
    description: "Aggregated scoring system to recommend the best products for you",
  },
  {
    icon: Zap,
    title: "Real-time Analysis",
    description: "Instant recommendations with detailed sentiment insights",
  },
];

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-hero">
      {/* Hero Section */}
      <section className="container px-4 py-16 mx-auto text-center">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-center mb-8">
            <div className="relative">
              <Brain className="h-16 w-16 text-primary animate-pulse" />
              <Sparkles className="absolute -top-2 -right-2 h-8 w-8 text-primary-glow animate-bounce" />
            </div>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold bg-gradient-primary bg-clip-text text-transparent mb-6">
            AI Product Recommendation System
          </h1>
          
          <p className="text-xl md:text-2xl text-muted-foreground mb-8 leading-relaxed">
            Discover the perfect products using advanced sentiment analysis and natural language processing.
            Just describe what you're looking for, and let our AI find the best matches.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button asChild size="lg" className="text-lg px-8 py-6 shadow-elevated hover:shadow-hover transition-all">
              <Link to="/query">
                Start Recommendation <Sparkles className="ml-2 h-5 w-5" />
              </Link>
            </Button>
            
            <Button asChild variant="outline" size="lg" className="text-lg px-8 py-6 hover:bg-card-hover">
              <Link to="/dashboard">
                View Dashboard
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="container px-4 py-16 mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">How It Works</h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Our AI-powered system combines multiple technologies to deliver accurate product recommendations
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <Card key={index} className="bg-gradient-card shadow-card hover:shadow-hover transition-all duration-300 border-0">
              <CardContent className="p-6 text-center">
                <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-4">
                  <feature.icon className="h-6 w-6 text-primary" />
                </div>
                <h3 className="font-semibold mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="container px-4 py-16 mx-auto">
        <Card className="bg-gradient-primary p-8 text-center border-0 shadow-elevated">
          <CardContent className="p-0">
            <h2 className="text-3xl font-bold text-primary-foreground mb-4">
              Ready to Find Your Perfect Product?
            </h2>
            <p className="text-primary-foreground/80 mb-6 text-lg">
              Experience the power of AI-driven recommendations tailored to your needs
            </p>
            <Button asChild size="lg" variant="secondary" className="text-lg px-8 py-6">
              <Link to="/query">
                Get Started Now
              </Link>
            </Button>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}