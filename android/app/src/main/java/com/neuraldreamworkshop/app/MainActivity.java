package com.neuraldreamworkshop.app;

import android.os.Bundle;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.Manifest;
import android.content.pm.PackageManager;
import androidx.core.content.ContextCompat;
import com.getcapacitor.BridgeActivity;

/**
 * Main activity — extends Capacitor's BridgeActivity.
 *
 * Capacitor's BridgeWebChromeClient already handles onPermissionRequest
 * and will prompt for RECORD_AUDIO / CAMERA via its own ActivityResultLauncher.
 *
 * We just:
 * 1. Disable MediaPlaybackRequiresUserGesture (for autoplay)
 * 2. Pre-grant MODIFY_AUDIO_SETTINGS if not already granted
 *    (Capacitor requests this alongside RECORD_AUDIO in onPermissionRequest)
 */
public class MainActivity extends BridgeActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // Register native Muse BLE plugin (must be before super.onCreate)
        registerPlugin(MuseBlePlugin.class);

        super.onCreate(savedInstanceState);

        WebView webView = getBridge().getWebView();
        WebSettings ws = webView.getSettings();
        ws.setMediaPlaybackRequiresUserGesture(false);
        ws.setDomStorageEnabled(true);
        ws.setJavaScriptEnabled(true);
    }
}
