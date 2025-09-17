import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { BarChart3, PieChart, TrendingUp, Users, Star, Package } from "lucide-react";
import { Link } from "react-router-dom";

// Mock dashboard data
const dashboardData = {
  totalQueries: 1247,
  totalProducts: 542,
  avgSentimentScore: 7.8,
  popularCategories: [
    { name: "Laptops", count: 342, percentage: 28 },
    { name: "Smartphones", count: 298, percentage: 24 },
    { name: "Headphones", count: 189, percentage: 15 },
    { name: "Smartwatches", count: 156, percentage: 13 },
    { name: "Other", count: 252, percentage: 20 },
  ],
  sentimentDistribution: [
    { sentiment: "Positive", count: 678, percentage: 54 },
    { sentiment: "Neutral", count: 398, percentage: 32 },
    { sentiment: "Negative", count: 171, percentage: 14 },
  ],
  recentQueries: [
    "Best laptop for programming with long battery life",
    "Smartphone with excellent camera under $800",
    "Wireless headphones for gym workouts",
    "Gaming laptop with RTX graphics",
    "Budget tablet for students",
  ],
};

export default function Dashboard() {
  const getSentimentColor = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'positive':
        return 'bg-sentiment-positive text-sentiment-positive-foreground';
      case 'neutral':
        return 'bg-sentiment-neutral text-sentiment-neutral-foreground';
      case 'negative':
        return 'bg-sentiment-negative text-sentiment-negative-foreground';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-hero py-8">
      <div className="container max-w-7xl mx-auto px-4">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4">Analytics Dashboard</h1>
          <p className="text-muted-foreground text-lg">
            Insights into product recommendations and sentiment analysis performance
          </p>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="bg-gradient-card shadow-card border-0">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Total Queries</p>
                  <p className="text-2xl font-bold text-primary">{dashboardData.totalQueries}</p>
                </div>
                <Users className="h-8 w-8 text-primary/60" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-card shadow-card border-0">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Products Analyzed</p>
                  <p className="text-2xl font-bold text-primary">{dashboardData.totalProducts}</p>
                </div>
                <Package className="h-8 w-8 text-primary/60" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-card shadow-card border-0">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Avg Sentiment</p>
                  <p className="text-2xl font-bold text-sentiment-positive">{dashboardData.avgSentimentScore}/10</p>
                </div>
                <Star className="h-8 w-8 text-sentiment-positive/60" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-card shadow-card border-0">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Success Rate</p>
                  <p className="text-2xl font-bold text-sentiment-positive">94%</p>
                </div>
                <TrendingUp className="h-8 w-8 text-sentiment-positive/60" />
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Category Distribution */}
          <Card className="bg-gradient-card shadow-card border-0">
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <PieChart className="h-5 w-5 text-primary" />
                Popular Categories
              </CardTitle>
              <Badge variant="secondary">{dashboardData.popularCategories.length} categories</Badge>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {dashboardData.popularCategories.map((category, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-3 h-3 rounded-full bg-primary" style={{
                        backgroundColor: `hsl(${260 + index * 20} 84% 60%)`
                      }} />
                      <span className="font-medium">{category.name}</span>
                    </div>
                    <div className="text-right">
                      <p className="font-bold">{category.count}</p>
                      <p className="text-xs text-muted-foreground">{category.percentage}%</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Sentiment Distribution */}
          <Card className="bg-gradient-card shadow-card border-0">
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-primary" />
                Sentiment Distribution
              </CardTitle>
              <Badge variant="secondary">Last 30 days</Badge>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {dashboardData.sentimentDistribution.map((item, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{item.sentiment}</span>
                      <Badge className={getSentimentColor(item.sentiment)}>
                        {item.count} ({item.percentage}%)
                      </Badge>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full transition-all duration-500 ${
                          item.sentiment === 'Positive' ? 'bg-sentiment-positive' :
                          item.sentiment === 'Neutral' ? 'bg-sentiment-neutral' : 'bg-sentiment-negative'
                        }`}
                        style={{ width: `${item.percentage}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Recent Queries */}
        <Card className="bg-gradient-card shadow-card border-0">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Recent User Queries</CardTitle>
            <Button asChild variant="outline" className="hover:bg-card-hover">
              <Link to="/query">
                Try New Query
              </Link>
            </Button>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3">
              {dashboardData.recentQueries.map((query, index) => (
                <div key={index} className="p-3 rounded-lg bg-muted/30 border border-border/50">
                  <p className="text-sm">"{query}"</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}