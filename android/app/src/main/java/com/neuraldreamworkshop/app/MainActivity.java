package com.neuraldreamworkshop.app;

import android.os.Bundle;
import android.webkit.PermissionRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.Manifest;
import android.content.pm.PackageManager;
import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.getcapacitor.BridgeActivity;

/**
 * Main activity — extends Capacitor's BridgeActivity.
 *
 * DO NOT call setWebChromeClient() — that replaces Capacitor's internal
 * BridgeWebChromeClient and breaks the JS bridge, dialogs, and file uploads.
 *
 * Instead, we request mic/camera permissions eagerly in onCreate so that by
 * the time the WebView fires getUserMedia, Android runtime permissions are
 * already granted and Capacitor's default handler can approve them.
 */
public class MainActivity extends BridgeActivity {

    private static final int PERMISSION_REQUEST_CODE = 1001;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Allow autoplay of audio/video without requiring a user gesture
        WebView webView = getBridge().getWebView();
        WebSettings ws = webView.getSettings();
        ws.setMediaPlaybackRequiresUserGesture(false);

        // Request mic + camera permissions upfront so that getUserMedia works
        // inside the WebView without hitting "Permission denied".
        requestMediaPermissionsIfNeeded();
    }

    private void requestMediaPermissionsIfNeeded() {
        boolean needsMic = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED;
        boolean needsCam = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED;

        if (needsMic || needsCam) {
            String[] perms;
            if (needsMic && needsCam) {
                perms = new String[]{ Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA };
            } else if (needsMic) {
                perms = new String[]{ Manifest.permission.RECORD_AUDIO };
            } else {
                perms = new String[]{ Manifest.permission.CAMERA };
            }
            ActivityCompat.requestPermissions(this, perms, PERMISSION_REQUEST_CODE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        // Capacitor's bridge handles the result propagation to plugins.
        // No additional handling needed here.
    }
}
