import { Routes } from '@angular/router';
import { DashboardLayout } from './components/dashboard-layout/dashboard-layout';

export const routes: Routes = [
  {
    path: '',
    component: DashboardLayout,
    title: "Tableau de bord — Progression de l'apprenant",
  },
  { path: '**', redirectTo: '' },
];
