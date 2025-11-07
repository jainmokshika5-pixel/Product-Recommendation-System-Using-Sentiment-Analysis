import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Search, Loader2, Lightbulb } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";

// Sample queries removed - using only real backend data

export default function Query() {
  const [query, setQuery] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [extractedFeatures, setExtractedFeatures] = useState<{
    category: string;
    features: string[];
  } | null>(null);
  
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!query.trim()) {
      toast({
        title: "Empty Query",
        description: "Please enter a product description to analyze.",
        variant: "destructive"
      });
      return;
    }

    setIsAnalyzing(true);
    
    try {
      // Call backend API for product recommendations
      const response = await fetch('http://localhost:8000/recommend_products', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query.trim() })
      });

      if (!response.ok) {
        throw new Error(`API call failed: ${response.status}`);
      }

      const data = await response.json();
      
      console.log('API Response:', data);
      console.log('Recommendations count:', data.recommendations?.length || 0);
      
      setExtractedFeatures(data.query_analysis);
      
      // Navigate to results after a brief delay
      setTimeout(() => {
        console.log('Navigating to results with state:', { 
          query, 
          analysis: data.query_analysis,
          recommendations: data.recommendations,
          totalProducts: data.total_products_analyzed
        });
        navigate('/results', { state: { 
          query, 
          analysis: data.query_analysis,
          recommendations: data.recommendations,
          totalProducts: data.total_products_analyzed
        } });
      }, 1000);
      
    } catch (error) {
      console.error('API Error:', error);
      
      toast({
        title: "Backend Error",
        description: "Unable to connect to the recommendation service. Please ensure the backend is running.",
        variant: "destructive"
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const extractFeaturesFromQuery = (text: string): string[] => {
    const features: string[] = [];
    const keywords = {
      'battery': ['battery', 'power', 'long-lasting'],
      'performance': ['fast', 'performance', 'speed', 'quick'],
      'camera': ['camera', 'photo', 'picture'],
      'display': ['screen', 'display', 'resolution'],
      'audio': ['sound', 'audio', 'music', 'noise'],
    };
    
    Object.entries(keywords).forEach(([feature, words]) => {
      if (words.some(word => text.toLowerCase().includes(word))) {
        features.push(feature);
      }
    });
    
    return features;
  };

  // Sample query handler removed - using only real backend data

  return (
    <div className="min-h-screen bg-gradient-hero py-8">
      <div className="container max-w-4xl mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4">What Are You Looking For?</h1>
          <p className="text-muted-foreground text-lg">
            Describe the product you want in natural language, and our AI will find the best matches
          </p>
        </div>

        <Card className="bg-gradient-card shadow-elevated border-0 mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="h-5 w-5 text-primary" />
              Product Query
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <Textarea
                placeholder="E.g., I need a laptop with excellent battery life, fast performance for programming, and a good display for design work..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="min-h-32 text-base resize-none border-2 focus:border-primary transition-colors"
                disabled={isAnalyzing}
              />
              
              <Button 
                type="submit" 
                size="lg" 
                className="w-full text-lg py-6 shadow-hover"
                disabled={isAnalyzing}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Analyzing Query...
                  </>
                ) : (
                  <>
                    <Search className="mr-2 h-5 w-5" />
                    Find Products
                  </>
                )}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Extracted Features Display */}
        {extractedFeatures && (
          <Card className="bg-gradient-card shadow-card border-0 mb-8">
            <CardContent className="p-6">
              <h3 className="font-semibold mb-4 flex items-center gap-2">
                <Lightbulb className="h-5 w-5 text-primary" />
                Analysis Results
              </h3>
              
              <div className="space-y-3">
                <div>
                  <span className="text-sm font-medium text-muted-foreground">Category: </span>
                  <Badge variant="secondary" className="bg-primary/10 text-primary">
                    {extractedFeatures.category}
                  </Badge>
                </div>
                
                <div>
                  <span className="text-sm font-medium text-muted-foreground">Features: </span>
                  <div className="flex flex-wrap gap-2 mt-1">
                    {extractedFeatures.features.map((feature, index) => (
                      <Badge key={index} variant="outline" className="border-primary/20">
                        {feature}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Sample queries section removed - using only real backend data */}
      </div>
    </div>
  );
}