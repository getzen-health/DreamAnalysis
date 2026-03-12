package com.neuraldreamworkshop.app;

import android.os.Bundle;
import android.webkit.PermissionRequest;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.Manifest;
import android.content.pm.PackageManager;
import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {

    private static final int PERMISSION_REQUEST_CODE = 1001;
    private PermissionRequest pendingPermissionRequest;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Request microphone + camera permissions upfront on launch
        String[] permissions = {
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.CAMERA
        };
        boolean needsRequest = false;
        for (String perm : permissions) {
            if (ContextCompat.checkSelfPermission(this, perm) != PackageManager.PERMISSION_GRANTED) {
                needsRequest = true;
                break;
            }
        }
        if (needsRequest) {
            ActivityCompat.requestPermissions(this, permissions, PERMISSION_REQUEST_CODE);
        }
    }

    @Override
    public void onStart() {
        super.onStart();

        WebView webView = getBridge().getWebView();

        // Configure WebView settings for media capture
        WebSettings webSettings = webView.getSettings();
        webSettings.setMediaPlaybackRequiresUserGesture(false);
        webSettings.setDomStorageEnabled(true);
        webSettings.setJavaScriptEnabled(true);

        // Override WebChromeClient to handle getUserMedia permission requests.
        // We only override onPermissionRequest; all other methods (alerts, file choosers,
        // console messages) inherit the default WebChromeClient behavior.
        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(final PermissionRequest request) {
                String[] resources = request.getResources();
                boolean needsAudio = false;
                boolean needsVideo = false;

                for (String resource : resources) {
                    if (PermissionRequest.RESOURCE_AUDIO_CAPTURE.equals(resource)) {
                        needsAudio = true;
                    }
                    if (PermissionRequest.RESOURCE_VIDEO_CAPTURE.equals(resource)) {
                        needsVideo = true;
                    }
                }

                // If the request is not for audio/video, deny to avoid granting unknown permissions
                if (!needsAudio && !needsVideo) {
                    request.deny();
                    return;
                }

                // Check if Android runtime permissions are already granted
                boolean audioGranted = !needsAudio ||
                    ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.RECORD_AUDIO)
                        == PackageManager.PERMISSION_GRANTED;
                boolean videoGranted = !needsVideo ||
                    ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA)
                        == PackageManager.PERMISSION_GRANTED;

                if (audioGranted && videoGranted) {
                    // Runtime permissions already granted — allow WebView access on UI thread
                    runOnUiThread(() -> request.grant(resources));
                } else {
                    // Need to request runtime permissions first — store pending request
                    pendingPermissionRequest = request;
                    String[] permsNeeded;
                    if (needsAudio && needsVideo) {
                        permsNeeded = new String[]{ Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA };
                    } else if (needsAudio) {
                        permsNeeded = new String[]{ Manifest.permission.RECORD_AUDIO };
                    } else {
                        permsNeeded = new String[]{ Manifest.permission.CAMERA };
                    }
                    ActivityCompat.requestPermissions(MainActivity.this, permsNeeded, PERMISSION_REQUEST_CODE);
                }
            }
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == PERMISSION_REQUEST_CODE && pendingPermissionRequest != null) {
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }
            final PermissionRequest req = pendingPermissionRequest;
            pendingPermissionRequest = null;

            if (allGranted) {
                runOnUiThread(() -> req.grant(req.getResources()));
            } else {
                runOnUiThread(req::deny);
            }
        }
    }
}
