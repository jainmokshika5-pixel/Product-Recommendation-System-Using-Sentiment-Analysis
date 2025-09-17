import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

interface SentimentBadgeProps {
  sentiment: "positive" | "neutral" | "negative";
  score: number;
  className?: string;
}

const getSentimentConfig = (sentiment: "positive" | "neutral" | "negative") => {
  switch (sentiment) {
    case "positive":
      return {
        className: "bg-sentiment-positive text-sentiment-positive-foreground hover:bg-sentiment-positive/90",
        label: "Positive",
      };
    case "neutral":
      return {
        className: "bg-sentiment-neutral text-sentiment-neutral-foreground hover:bg-sentiment-neutral/90",
        label: "Neutral",
      };
    case "negative":
      return {
        className: "bg-sentiment-negative text-sentiment-negative-foreground hover:bg-sentiment-negative/90",
        label: "Negative",
      };
  }
};

export function SentimentBadge({ sentiment, score, className }: SentimentBadgeProps) {
  const config = getSentimentConfig(sentiment);
  
  return (
    <Badge
      variant="secondary"
      className={cn(config.className, "font-medium", className)}
    >
      {config.label} {score.toFixed(1)}
    </Badge>
  );
}