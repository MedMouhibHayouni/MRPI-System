export type MissionStatus = 'not_started' | 'in_progress' | 'completed';

/** Ordered mastery ladder shown per subject: Novice → Apprenti → Praticien → Expert. */
export type MasteryLevel = 'Novice' | 'Apprenti' | 'Praticien' | 'Expert';

/** Derives the mastery level label from a 0..1 completion percentage. */
export function getMasteryLevel(percentage: number): MasteryLevel {
  if (percentage < 0.25) return 'Novice';
  if (percentage < 0.5) return 'Apprenti';
  if (percentage < 0.75) return 'Praticien';
  return 'Expert';
}

export interface SubjectMastery {
  subject: string;
  xp: number;
  xp_next: number;
  percentage: number; // 0..1
}

export interface WeeklyMission {
  id: string;
  title: string;
  status: MissionStatus;
  deadline: string; // ISO date
}

export interface ProgressResponse {
  student_id: string;
  subject_mastery: SubjectMastery[];
  weekly_missions: WeeklyMission[];
}
