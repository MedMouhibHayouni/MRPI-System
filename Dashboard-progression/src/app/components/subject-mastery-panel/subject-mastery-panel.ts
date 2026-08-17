import { Component, computed, input } from '@angular/core';
import { LucideAngularModule } from 'lucide-angular';
import { SubjectMastery } from '../../models/progress.model';
import { XpProgressBar } from '../xp-progress-bar/xp-progress-bar';

@Component({
  selector: 'app-subject-mastery-panel',
  standalone: true,
  imports: [LucideAngularModule, XpProgressBar],
  templateUrl: './subject-mastery-panel.html',
  styleUrl: './subject-mastery-panel.css',
})
export class SubjectMasteryPanel {
  items = input.required<SubjectMastery[]>();

  avg = computed(() =>
    Math.round((this.items().reduce((s, i) => s + i.percentage, 0) / this.items().length) * 100),
  );
}
