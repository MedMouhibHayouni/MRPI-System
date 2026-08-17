/** A single earned badge: a short label plus a lucide-angular icon name (kebab-case). */
export interface Badge {
  id: string;
  label: string;
  icon: string;
}

export interface StudentProfile {
  id: string;
  name: string;
  level: number;
  avatar_url: string;
  total_xp: number;
  badges?: Badge[];
}
