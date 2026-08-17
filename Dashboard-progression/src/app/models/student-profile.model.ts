import { AcademicLevel, LearningStyle } from './recommendation.model';

/**
 * Real learner profile as it exists in students.csv (the dataset the FastAPI
 * backend was trained/tested against). This is exactly what fuels the CBF+CF
 * hybrid engine — the student_id alone is not enough input for
 * POST /recommendations, it needs subject/weak_concept/academic_level/
 * learning_style/past_interactions.
 *
 * The backend has no GET-by-id endpoint that returns this profile
 * (/student-profile only validates one you already have and echoes it back),
 * so it's bundled as a static asset (public/students.json) rather than
 * invented per component. main.py is not being modified.
 */
export interface StudentAcademicProfile {
  student_id: string;
  subject: string;
  weak_concept: string;
  academic_level: AcademicLevel;
  learning_style: LearningStyle;
  past_interactions: string[];
}
