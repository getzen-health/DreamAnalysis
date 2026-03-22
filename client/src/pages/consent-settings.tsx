/**
 * Granular biometric consent settings page.
 *
 * GDPR / HIPAA compliance: per-modality consent toggles.
 * All modalities default to OFF — no pre-checked boxes.
 * Accept All and Reject All have EQUAL visual weight.
 */

import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Shield } from "lucide-react";
import { useConsent } from "@/hooks/use-consent";
import { CONSENT_MODALITIES, type ConsentModality } from "@/lib/consent-store";

export default function ConsentSettings() {
  const consent = useConsent();

  return (
    <main className="p-4 md:p-6 pb-24 space-y-6 max-w-3xl">
      {/* Header */}
      <div className="flex items-start gap-4">
        <div
          className="w-12 h-12 rounded-xl flex items-center justify-center shrink-0 mt-1"
          style={{
            background: "hsl(152, 60%, 48%, 0.12)",
            border: "1px solid hsl(152, 60%, 48%, 0.3)",
          }}
        >
          <Shield className="h-6 w-6 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-semibold">Biometric Consent</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Control which types of biometric data the app can collect and process.
            All toggles are off by default. You can change these at any time.
          </p>
        </div>
      </div>

      {/* Accept All / Reject All — EQUAL visual weight */}
      <div className="flex gap-3">
        <Button
          variant="outline"
          className="flex-1 border-primary/30 text-primary hover:bg-primary/10"
          onClick={consent.acceptAll}
        >
          Accept All
        </Button>
        <Button
          variant="outline"
          className="flex-1 border-primary/30 text-primary hover:bg-primary/10"
          onClick={consent.rejectAll}
        >
          Reject All
        </Button>
      </div>

      {/* Per-modality toggles */}
      <Card className="glass-card p-5 space-y-1">
        {CONSENT_MODALITIES.map((modality) => (
          <div
            key={modality.id}
            className="flex items-center justify-between py-3 border-b border-border/20 last:border-b-0"
          >
            <div className="flex-1 min-w-0 pr-4">
              <Label className="text-sm font-medium">{modality.label}</Label>
              <p className="text-xs text-muted-foreground mt-0.5">
                {modality.description}
              </p>
            </div>
            <Switch
              checked={consent[modality.id]}
              onCheckedChange={(checked) => consent.toggle(modality.id, checked)}
              data-testid={`consent-toggle-${modality.id}`}
            />
          </div>
        ))}
      </Card>

      {/* Info notice */}
      <div className="rounded-lg border border-border/40 bg-muted/20 p-4 text-xs text-muted-foreground space-y-2">
        <p>
          Your consent choices are stored locally on your device and synced to your
          account when connected. You can withdraw consent at any time by toggling
          a category off.
        </p>
        <p>
          When a category is off, the app will not collect, process, or store that
          type of data. Previously collected data can be deleted from Settings &gt;
          Data &amp; Privacy.
        </p>
      </div>
    </main>
  );
}
