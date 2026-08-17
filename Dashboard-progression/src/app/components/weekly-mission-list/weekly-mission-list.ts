import { Component, computed, input } from '@angular/core';
import { LucideAngularModule } from 'lucide-angular';
import { WeeklyMission } from '../../models/progress.model';
import { WeeklyMissionItem } from '../weekly-mission-item/weekly-mission-item';

@Component({
  selector: 'app-weekly-mission-list',
  standalone: true,
  imports: [LucideAngularModule, WeeklyMissionItem],
  templateUrl: './weekly-mission-list.html',
  styleUrl: './weekly-mission-list.css',
})
export class WeeklyMissionList {
  missions = input.required<WeeklyMission[]>();

  done = computed(() => this.missions().filter((m) => m.status === 'completed').length);
  pct = computed(() => Math.round((this.done() / this.missions().length) * 100));
}
