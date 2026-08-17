import { Injectable } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';

/**
 * Centralized error handling for all HTTP calls, per 4.4.3.
 * Every service maps its HttpErrorResponse through this single
 * point so user-facing messages stay consistent app-wide.
 */
@Injectable({ providedIn: 'root' })
export class ApiErrorHandler {
  toUserMessage(error: unknown): string {
    if (error instanceof HttpErrorResponse) {
      if (error.status === 0) {
        return 'Impossible de contacter le serveur. Vérifie ta connexion.';
      }
      if (error.status === 404) {
        return 'Ressource introuvable.';
      }
      if (error.status >= 500) {
        return 'Le serveur rencontre un problème. Réessaie dans un instant.';
      }
      return `Une erreur est survenue (code ${error.status}).`;
    }
    return 'Une erreur inattendue est survenue.';
  }
}
