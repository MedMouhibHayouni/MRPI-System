import { Component, computed, input } from '@angular/core';
import { getMasteryLevel, MasteryLevel, SubjectMastery } from '../../models/progress.model';

@Component({
  selector: 'app-xp-progress-bar',
  standalone: true,
  templateUrl: './xp-progress-bar.html',
  styleUrl: './xp-progress-bar.css',
})
export class XpProgressBar {
  item = input.required<SubjectMastery>();

  pct = computed(() => Math.round(this.item().percentage * 100));

  /** Mastery ladder position: Novice → Apprenti → Praticien → Expert. */
  level = computed<MasteryLevel>(() => getMasteryLevel(this.item().percentage));

  /** Tailwind classes color-coding the level pill, from neutral (Novice) to success (Expert). */
  levelClass = computed(() => {
    switch (this.level()) {
      case 'Novice':
        return 'bg-muted text-muted-foreground';
      case 'Apprenti':
        return 'bg-warning/15 text-warning';
      case 'Praticien':
        return 'bg-primary/15 text-primary';
      case 'Expert':
        return 'bg-success/15 text-success';
    }
  });
}
