import { Component, computed, input } from '@angular/core';
import { NgClass } from '@angular/common';
import { LucideAngularModule } from 'lucide-angular';
import { MissionStatus, WeeklyMission } from '../../models/progress.model';

interface StatusMeta {
  label: string;
  iconCls: string;
  badgeCls: string;
}

const STATUS_META: Record<MissionStatus, StatusMeta> = {
  not_started: { label: 'À faire', iconCls: 'bg-muted text-muted-foreground', badgeCls: 'bg-muted text-muted-foreground' },
  in_progress: { label: 'En cours', iconCls: 'bg-primary/15 text-primary', badgeCls: 'bg-primary/15 text-primary' },
  completed: { label: 'Terminée', iconCls: 'bg-success/20 text-success', badgeCls: 'bg-success/20 text-success' },
};

@Component({
  selector: 'app-weekly-mission-item',
  standalone: true,
  imports: [NgClass, LucideAngularModule],
  templateUrl: './weekly-mission-item.html',
  styleUrl: './weekly-mission-item.css',
})
export class WeeklyMissionItem {
  mission = input.required<WeeklyMission>();

  statusMeta = computed(() => STATUS_META[this.mission().status]);
  isDone = computed(() => this.mission().status === 'completed');
  isInProgress = computed(() => this.mission().status === 'in_progress');

  formattedDeadline = computed(() => {
    const d = new Date(this.mission().deadline);
    return d.toLocaleDateString('fr-FR', { day: '2-digit', month: 'short' });
  });
}
