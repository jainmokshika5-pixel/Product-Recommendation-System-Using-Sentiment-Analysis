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
  const location = useLocation();
  const state = location.state as LocationState | null;

  if (!state) {
    return <Navigate to="/query" replace />;
  }

  const { query, analysis, recommendations, totalProducts } = state;

  // Use real API data if available, otherwise fallback to mock data
  const products = recommendations || [
    {
      id: 1,
      name: "Apple iPhone 15 Pro (128GB) - Natural Titanium",
      category: "Smartphone",
      price: 134900,
      originalPrice: 139900,
      description: "A17 Pro chip with 6-core GPU, Advanced camera system, Action Button",
      sentiment_score: 8.5,
      sentiment_label: "positive" as const,
      explanation: "Highly rated for camera and performance",
      rating: 4.4,
      reviewCount: 2847,
      deliveryInfo: "FREE delivery by Tomorrow",
      offers: ["Bank Offer", "Exchange Offer"],
      image: "https://images.unsplash.com/photo-1592750475338-74b7b21085ab?w=300&h=300&fit=crop"
    },
    {
      id: 2,
      name: "Apple MacBook Pro (14-inch, M3, 8GB RAM, 512GB SSD)",
      category: "Laptop", 
      price: 169900,
      originalPrice: 199900,
      description: "M3 chip with 8-core CPU, 10-core GPU, 14-inch Liquid Retina XDR display",
      sentiment_score: 9.1,
      sentiment_label: "positive" as const,
      explanation: "Excellent performance and battery life",
      rating: 4.6,
      reviewCount: 1923,
      deliveryInfo: "FREE delivery by Tomorrow",
      offers: ["Bank Offer", "No Cost EMI"],
      image: "https://images.unsplash.com/photo-1541807084-5c52b6b3adef?w=300&h=300&fit=crop"
    },
    {
      id: 3,
      name: "Samsung Galaxy S24+ 5G (256GB) - Onyx Black",
      category: "Smartphone",
      price: 89999,
      originalPrice: 99999,
      description: "50MP Triple Camera, 6.7-inch Dynamic AMOLED 2X Display, 4900mAh Battery",
      sentiment_score: 8.2,
      sentiment_label: "positive" as const,
      explanation: "Great display and camera quality",
      rating: 4.3,
      reviewCount: 3521,
      deliveryInfo: "FREE delivery by Tomorrow",
      offers: ["Exchange Offer", "Bank Offer"],
      image: "https://images.unsplash.com/photo-1610945265064-0e34e5519bbf?w=300&h=300&fit=crop"
    }
  ];

  const getPriceColor = (price: number) => {
    if (price < 50000) return "text-sentiment-positive";
    if (price < 100000) return "text-sentiment-neutral"; 
    return "text-sentiment-negative";
  };

  const formatPrice = (price: number) => {
    return `₹${price.toLocaleString('en-IN')}`;
  };

  const calculateDiscount = (price: number, originalPrice: number) => {
    return Math.round(((originalPrice - price) / originalPrice) * 100);
  };

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
        <div className="grid gap-4">
          {products.map((product, index) => (
            <Card key={product.id} className="bg-gradient-card shadow-card hover:shadow-hover transition-all border-0 overflow-hidden">
              <div className="flex gap-4 p-4">
                {/* Product Image */}
                <div className="flex-shrink-0">
                  <div className="w-32 h-32 bg-muted rounded-lg overflow-hidden">
                    <img 
                      src={product.image} 
                      alt={product.name}
                      className="w-full h-full object-cover"
                    />
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
                  
                  {/* Rating and Reviews */}
                  <div className="flex items-center gap-3 mb-2">
                    <div className="flex items-center gap-1">
                      <div className="flex items-center">
                        <span className="text-sm font-medium text-white bg-green-600 px-2 py-1 rounded">
                          {product.rating} ★
                        </span>
                      </div>
                      <span className="text-sm text-muted-foreground">
                        ({product.reviewCount.toLocaleString()} reviews)
                      </span>
                    </div>
                  </div>
                  
                  {/* Price Section */}
                  <div className="flex items-center gap-3 mb-2">
                    <span className={`text-2xl font-bold ${getPriceColor(product.price)}`}>
                      {formatPrice(product.price)}
                    </span>
                    {product.originalPrice && product.originalPrice > product.price && (
                      <>
                        <span className="text-sm text-muted-foreground line-through">
                          {formatPrice(product.originalPrice)}
                        </span>
                        <span className="text-sm font-medium text-green-600">
                          {calculateDiscount(product.price, product.originalPrice)}% off
                        </span>
                      </>
                    )}
                  </div>
                  
                  {/* Delivery Info */}
                  <p className="text-sm text-green-600 font-medium mb-2">
                    {product.deliveryInfo}
                  </p>
                  
                  {/* Offers */}
                  <div className="flex flex-wrap gap-2 mb-3">
                    {product.offers.map((offer, index) => (
                      <Badge key={index} variant="outline" className="text-xs text-orange-600 border-orange-200">
                        {offer}
                      </Badge>
                    ))}
                  </div>
                  
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
}