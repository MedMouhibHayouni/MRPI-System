export type AcademicLevel = 'beginner' | 'intermediate' | 'advanced';
// 'textual' included because it exists as a real value in students.csv (10/41 rows).
export type LearningStyle = 'visual' | 'auditory' | 'kinesthetic' | 'textual';

/** Request body for POST /recommendations, per Tableau 6 (schema d'entree - profil apprenant). */
export interface RecommendationRequest {
  student_id: string;
  subject: string;
  weak_concept: string;
  academic_level: AcademicLevel;
  learning_style: LearningStyle;
  past_interactions?: string[];
}

export type RecommendationType = 'exercise' | 'micro_lesson' | 'video';
export type Difficulty = 'beginner' | 'intermediate' | 'advanced';

export interface Recommendation {
  resource_id: string;
  title: string;
  type: RecommendationType;
  relevance_score: number; // 0..1
  difficulty: Difficulty;
  estimated_time_min: number;
}

export interface RecommendationsResponse {
  request_id: string;
  student_id: string;
  recommendations: Recommendation[];
  adapted_recommendations?: unknown;
  study_plan?: unknown;
  llm_explanation: string;
  metadata: {
    latency_ms: number;
    [key: string]: unknown;
  };
}
