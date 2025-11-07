import { useLocation, Navigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SentimentBadge } from "@/components/ui/sentiment-badge";
import { ExternalLink, BarChart3, Lightbulb } from "lucide-react";
import { Link } from "react-router-dom";

interface Product {
  id: number;
  name: string;
  category: string;
  price: number;
  originalPrice?: number;
  description: string;
  sentiment_score: number;
  sentiment_label: "positive" | "neutral" | "negative";
  explanation: string;
  rating?: number;
  reviewCount?: number;
  deliveryInfo?: string;
  offers?: string[];
  image?: string;
}

interface LocationState {
  query: string;
  analysis: {
    category: string;
    features: string[];
    confidence: number;
  };
  recommendations?: Product[];
  totalProducts?: number;
}

export default function Results() {
  console.log('Results component rendering...');
  
  const location = useLocation();
  const state = location.state as LocationState | null;

  console.log('Results page loaded, location:', location);
  console.log('Results page state:', state);

  if (!state) {
    console.log('No state found, redirecting to query page');
    return (
      <div className="min-h-screen bg-gradient-hero py-8">
        <div className="container max-w-6xl mx-auto px-4">
          <Card className="bg-gradient-card shadow-card border-0 p-8 text-center">
            <h2 className="text-xl font-semibold mb-4">No Search Results</h2>
            <p className="text-muted-foreground mb-4">
              Please go back to the search page and try again.
            </p>
            <Button asChild variant="outline">
              <Link to="/query">
                Back to Search
              </Link>
            </Button>
          </Card>
        </div>
      </div>
    );
  }

  const { query, analysis, recommendations, totalProducts } = state;

  // Use only real API data from backend
  const products = recommendations || [];
  
  // Debug logging
  console.log('Results page state:', { query, analysis, recommendations, totalProducts });
  console.log('Products count:', products.length);

  const getPriceColor = (price: number | undefined) => {
    if (!price) return "text-muted-foreground";
    if (price < 50000) return "text-sentiment-positive";
    if (price < 100000) return "text-sentiment-neutral"; 
    return "text-sentiment-negative";
  };

  const formatPrice = (price: number | undefined) => {
    if (!price) return 'Price not available';
    return `₹${price.toLocaleString('en-IN')}`;
  };

  const calculateDiscount = (price: number | undefined, originalPrice: number | undefined) => {
    if (!price || !originalPrice || originalPrice <= price) return 0;
    return Math.round(((originalPrice - price) / originalPrice) * 100);
  };

  try {
    return (
      <div className="min-h-screen bg-gradient-hero py-8">
        <div className="container max-w-6xl mx-auto px-4">
        {/* Query Summary */}
        <Card className="bg-gradient-card shadow-card border-0 mb-8">
          <CardContent className="p-6">
            <h1 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <Lightbulb className="h-6 w-6 text-primary" />
              Search Results
            </h1>
            
            <div className="bg-muted/50 rounded-lg p-4 mb-4">
              <p className="text-sm font-medium text-muted-foreground mb-2">Your Query:</p>
              <p className="text-base italic">"{query}"</p>
            </div>
            
            <div className="flex flex-wrap gap-4">
              <div>
                <span className="text-sm font-medium text-muted-foreground">Category: </span>
                <Badge variant="secondary" className="bg-primary/10 text-primary">
                  {analysis.category}
                </Badge>
              </div>
              
              <div>
                <span className="text-sm font-medium text-muted-foreground">Features: </span>
                <div className="inline-flex gap-2 ml-1">
                  {analysis.features.map((feature, index) => (
                    <Badge key={index} variant="outline" className="border-primary/20">
                      {feature}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Results Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold">
            Top {products.length} Recommended Products
            {totalProducts && (
              <span className="text-sm font-normal text-muted-foreground ml-2">
                (from {totalProducts} analyzed)
              </span>
            )}
          </h2>
          
          <Button asChild variant="outline" className="hover:bg-card-hover">
            <Link to="/dashboard">
              <BarChart3 className="mr-2 h-4 w-4" />
              View Analytics
            </Link>
          </Button>
        </div>

        {/* Products Grid */}
        {products.length > 0 ? (
          <div className="grid gap-4">
            {products.map((product, index) => (
            <Card key={product.id} className="bg-gradient-card shadow-card hover:shadow-hover transition-all border-0 overflow-hidden">
              <div className="flex gap-4 p-4">
                {/* Product Image */}
                <div className="flex-shrink-0">
                  <div className="w-32 h-32 bg-muted rounded-lg overflow-hidden">
                    {product.image ? (
                      <img 
                        src={product.image} 
                        alt={product.name}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-muted-foreground">
                        No Image
                      </div>
                    )}
                  </div>
                </div>
                
                {/* Product Details */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className="text-xs">
                        #{index + 1} Choice
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {product.category}
                      </Badge>
                    </div>
                    <SentimentBadge 
                      sentiment={product.sentiment_label}
                      score={product.sentiment_score}
                    />
                  </div>
                  
                  <h3 className="text-lg font-semibold mb-2 line-clamp-2 leading-tight">
                    {product.name}
                  </h3>
                  
                  {/* Sentiment Score */}
                  <div className="flex items-center gap-3 mb-2">
                    <div className="flex items-center gap-1">
                      <div className="flex items-center">
                        <span className="text-sm font-medium text-white bg-green-600 px-2 py-1 rounded">
                          {product.sentiment_score?.toFixed(1) || 'N/A'} ★
                        </span>
                      </div>
                      <span className="text-sm text-muted-foreground">
                        {product.sentiment_label || 'neutral'} sentiment
                      </span>
                    </div>
                  </div>
                  
                  {/* Price Section */}
                  <div className="flex items-center gap-3 mb-2">
                    <span className={`text-2xl font-bold ${getPriceColor(product.price)}`}>
                      {formatPrice(product.price)}
                    </span>
                  </div>
                  
                  {/* Recommendation Explanation */}
                  <p className="text-sm text-blue-600 font-medium mb-2">
                    {product.explanation}
                  </p>
                  
                  {/* Description */}
                  <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
                    {product.description}
                  </p>
                  
                  {/* Why Recommended */}
                  {product.explanation && (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-3">
                      <p className="text-sm text-blue-800">
                        <strong>Why recommended:</strong> {product.explanation}
                      </p>
                    </div>
                  )}
                  
                  {/* Features */}
                  <div className="flex flex-wrap gap-2 mb-4">
                    {analysis.features.slice(0, 3).map((feature, index) => (
                      <Badge key={index} variant="secondary" className="text-xs bg-accent/50">
                        {feature}
                      </Badge>
                    ))}
                  </div>
                </div>
                
                {/* Action Buttons */}
                <div className="flex flex-col gap-2 justify-center">
                  <Button className="bg-orange-500 hover:bg-orange-600 text-white font-semibold px-6">
                    Add to Cart
                  </Button>
                  <Button variant="outline" className="border-orange-500 text-orange-500 hover:bg-orange-50 font-semibold px-6">
                    Buy Now
                  </Button>
                  <Button variant="ghost" size="sm" className="text-xs">
                    <BarChart3 className="h-3 w-3 mr-1" />
                    Analytics
                  </Button>
                </div>
              </div>
            </Card>
          ))}
          </div>
        ) : (
          <Card className="bg-gradient-card shadow-card border-0 p-8 text-center">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-muted-foreground">No Products Found</h3>
              <p className="text-muted-foreground">
                We couldn't find any products matching your query. Try adjusting your search terms.
              </p>
              <Button asChild variant="outline">
                <Link to="/query">
                  Try Another Search
                </Link>
              </Button>
            </div>
          </Card>
        )}

        {/* Back to Search */}
        <div className="text-center mt-8">
          <Button asChild variant="outline" size="lg" className="hover:bg-card-hover">
            <Link to="/query">
              Search Again
            </Link>
          </Button>
        </div>
      </div>
    </div>
    );
  } catch (error) {
    console.error('Error in Results component:', error);
    return (
      <div className="min-h-screen bg-gradient-hero py-8">
        <div className="container max-w-6xl mx-auto px-4">
          <Card className="bg-gradient-card shadow-card border-0 p-8 text-center">
            <h2 className="text-xl font-semibold mb-4">Error Loading Results</h2>
            <p className="text-muted-foreground mb-4">
              There was an error loading the results. Please try again.
            </p>
            <Button asChild variant="outline">
              <Link to="/query">
                Back to Search
              </Link>
            </Button>
          </Card>
        </div>
      </div>
    );
  }
}