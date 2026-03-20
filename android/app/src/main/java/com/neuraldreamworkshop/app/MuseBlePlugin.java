package com.neuraldreamworkshop.app;

import android.Manifest;
import android.bluetooth.*;
import android.bluetooth.le.*;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.*;
import android.util.Base64;
import android.util.Log;
import androidx.core.app.ActivityCompat;
import com.getcapacitor.*;
import com.getcapacitor.annotation.*;
import org.json.JSONArray;
import org.json.JSONObject;
import java.lang.reflect.Method;
import java.util.*;

/**
 * Native Android BLE plugin for Muse headbands.
 * Uses BluetoothGatt directly with gatt.refresh() to fix GATT caching on Android 16.
 */
@CapacitorPlugin(
    name = "MuseBle",
    permissions = {
        @Permission(strings = {Manifest.permission.BLUETOOTH_SCAN}, alias = "bluetooth_scan"),
        @Permission(strings = {Manifest.permission.BLUETOOTH_CONNECT}, alias = "bluetooth_connect"),
        @Permission(strings = {Manifest.permission.ACCESS_FINE_LOCATION}, alias = "location"),
    }
)
public class MuseBlePlugin extends Plugin {
    private static final String TAG = "MuseBle";

    // Muse GATT UUIDs
    private static final UUID MUSE_SERVICE = UUID.fromString("0000fe8d-0000-1000-8000-00805f9b34fb");
    private static final UUID CONTROL_CHAR = UUID.fromString("273e0001-4c4d-454d-96be-f03bac821358");
    // Muse 2 EEG characteristic UUIDs
    private static final UUID[] EEG_CHARS_MUSE2 = {
        UUID.fromString("273e0003-4c4d-454d-96be-f03bac821358"),
        UUID.fromString("273e0004-4c4d-454d-96be-f03bac821358"),
        UUID.fromString("273e0005-4c4d-454d-96be-f03bac821358"),
        UUID.fromString("273e0006-4c4d-454d-96be-f03bac821358"),
    };
    // Muse S EEG characteristic UUIDs (offset by 0x10)
    private static final UUID[] EEG_CHARS_MUSE_S = {
        UUID.fromString("273e0013-4c4d-454d-96be-f03bac821358"),
        UUID.fromString("273e0014-4c4d-454d-96be-f03bac821358"),
        UUID.fromString("273e0015-4c4d-454d-96be-f03bac821358"),
        UUID.fromString("273e0016-4c4d-454d-96be-f03bac821358"),
    };
    // Active set — determined during service discovery
    private UUID[] EEG_CHARS = EEG_CHARS_MUSE2;
    private static final UUID CCC_DESCRIPTOR = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb");

    // Muse commands
    private static final byte[] CMD_PRESET = {0x04, 0x70, 0x32, 0x31, 0x0a};
    private static final byte[] CMD_START = {0x02, 0x64, 0x0a};
    private static final byte[] CMD_STOP = {0x02, 0x68, 0x0a};

    private final Handler handler = new Handler(Looper.getMainLooper());
    private BluetoothGatt bluetoothGatt;
    private PluginCall connectCall;
    private boolean isStreaming = false;
    private int subscribedChannels = 0;
    private int pendingDescriptorWrites = 0;
    private boolean commandsSent = false;
    private Runnable timeoutRunnable = null; // cancel previous timeouts

    @PluginMethod
    public void scan(PluginCall call) {
        BluetoothAdapter adapter = getAdapter();
        if (adapter == null) { call.reject("Bluetooth not available"); return; }
        if (!hasBlePermissions()) { requestAllPermissions(call, "permCallback"); return; }

        BluetoothLeScanner scanner = adapter.getBluetoothLeScanner();
        if (scanner == null) { call.reject("BLE scanner not available"); return; }

        List<JSONObject> devices = new ArrayList<>();

        ScanCallback scanCb = new ScanCallback() {
            @Override
            public void onScanResult(int callbackType, ScanResult result) {
                try {
                    String name = result.getDevice().getName();
                    if (name == null && result.getScanRecord() != null) name = result.getScanRecord().getDeviceName();
                    if (name == null) name = "Unknown";
                    if (!name.toLowerCase().contains("muse")) return;

                    String addr = result.getDevice().getAddress();
                    for (JSONObject d : devices) {
                        if (d.getString("deviceId").equals(addr)) return;
                    }
                    JSONObject d = new JSONObject();
                    d.put("deviceId", addr);
                    d.put("name", name);
                    d.put("rssi", result.getRssi());
                    devices.add(d);
                } catch (Exception ignored) {}
            }
        };

        ScanSettings settings = new ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build();

        // Scan WITHOUT service filter — Muse S may not always advertise service UUID
        try {
            scanner.startScan(null, settings, scanCb);
        } catch (SecurityException e) {
            call.reject("Scan permission denied");
            return;
        }

        // Scan for 12 seconds (longer to catch Muse S advertising cycle)
        handler.postDelayed(() -> {
            try { scanner.stopScan(scanCb); } catch (Exception ignored) {}
            JSObject result = new JSObject();
            JSONArray arr = new JSONArray();
            for (JSONObject d : devices) arr.put(d);
            result.put("devices", arr);
            call.resolve(result);
        }, 12000);
    }

    @PluginMethod
    public void connect(PluginCall call) {
        String deviceId = call.getString("deviceId");
        if (deviceId == null) { call.reject("deviceId required"); return; }
        if (!hasBlePermissions()) { requestAllPermissions(call, "permCallback"); return; }

        BluetoothAdapter adapter = getAdapter();
        if (adapter == null) { call.reject("Bluetooth not available"); return; }

        // Cancel any pending timeout from previous attempt
        if (timeoutRunnable != null) {
            handler.removeCallbacks(timeoutRunnable);
            timeoutRunnable = null;
        }

        // Cleanup existing connection
        if (bluetoothGatt != null) {
            try { bluetoothGatt.disconnect(); } catch (Exception ignored) {}
            try { bluetoothGatt.close(); } catch (Exception ignored) {}
            bluetoothGatt = null;
        }
        isStreaming = false;
        subscribedChannels = 0;
        commandsSent = false;
        connectCall = null; // clear any stale call

        BluetoothDevice device;
        try {
            device = adapter.getRemoteDevice(deviceId);
        } catch (Exception e) {
            call.reject("Invalid device: " + deviceId);
            return;
        }

        connectCall = call;

        BluetoothGattCallback gattCb = new BluetoothGattCallback() {
            @Override
            public void onConnectionStateChange(BluetoothGatt gatt, int status, int newState) {
                Log.d(TAG, "State: " + newState + " status: " + status);
                if (newState == BluetoothProfile.STATE_CONNECTED && status == BluetoothGatt.GATT_SUCCESS) {
                    Log.d(TAG, "Connected! Refreshing GATT cache...");
                    refreshGattCache(gatt);
                    handler.postDelayed(() -> {
                        try {
                            Log.d(TAG, "Discovering services...");
                            gatt.discoverServices();
                        } catch (SecurityException e) {
                            rejectConnect("Permission denied: " + e.getMessage());
                        }
                    }, 1500);
                } else if (newState == BluetoothProfile.STATE_CONNECTED && status != BluetoothGatt.GATT_SUCCESS) {
                    // Connected but with error status — try discovery anyway
                    Log.w(TAG, "Connected with non-zero status " + status + ", trying discovery...");
                    refreshGattCache(gatt);
                    handler.postDelayed(() -> {
                        try { gatt.discoverServices(); } catch (SecurityException ignored) {}
                    }, 2000);
                } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                    isStreaming = false;
                    if (status == 8 && !commandsSent) {
                        // GATT_CONN_TIMEOUT on first attempt — retry with direct connect
                        Log.w(TAG, "GATT timeout, retrying direct connect...");
                        try { gatt.close(); } catch (Exception ignored) {}
                        handler.postDelayed(() -> {
                            try {
                                bluetoothGatt = device.connectGatt(
                                    getContext(), false, this,
                                    BluetoothDevice.TRANSPORT_LE
                                );
                            } catch (SecurityException ignored) {
                                rejectConnect("Retry connect permission denied");
                            }
                        }, 2000);
                    } else {
                        notifyListeners("museDisconnected", new JSObject());
                        rejectConnect("Disconnected (status: " + status + ")");
                    }
                }
            }

            @Override
            public void onServicesDiscovered(BluetoothGatt gatt, int status) {
                Log.d(TAG, "Services discovered: " + status + " count: " + gatt.getServices().size() + " commandsSent: " + commandsSent);
                if (status != BluetoothGatt.GATT_SUCCESS) {
                    rejectConnect("Service discovery failed");
                    return;
                }

                BluetoothGattService svc = gatt.getService(MUSE_SERVICE);
                if (svc == null) {
                    Log.e(TAG, "Muse service NOT found. Services:");
                    for (BluetoothGattService s : gatt.getServices()) {
                        Log.e(TAG, "  " + s.getUuid());
                    }
                    rejectConnect("Muse service not found (" + gatt.getServices().size() + " services discovered)");
                    return;
                }

                // Log all characteristics in the service
                Log.d(TAG, "Muse service has " + svc.getCharacteristics().size() + " characteristics:");
                for (BluetoothGattCharacteristic c : svc.getCharacteristics()) {
                    Log.d(TAG, "  Char: " + c.getUuid() + " props: " + c.getProperties());
                }

                if (!commandsSent) {
                    // FIRST discovery: send preset + start commands, then re-discover
                    commandsSent = true;
                    handler.postDelayed(() -> {
                        BluetoothGattCharacteristic ctrl = svc.getCharacteristic(CONTROL_CHAR);
                        if (ctrl == null) {
                            rejectConnect("Control characteristic not found");
                            return;
                        }
                        writeChar(gatt, ctrl, CMD_PRESET);
                        handler.postDelayed(() -> {
                            writeChar(gatt, ctrl, CMD_START);
                            // Re-discover after Muse reconfigures its GATT table
                            handler.postDelayed(() -> {
                                Log.d(TAG, "Re-discovering services after preset+start...");
                                try { gatt.discoverServices(); } catch (SecurityException ignored) {}
                            }, 2000);
                        }, 500);
                    }, 500);
                } else {
                    // SECOND discovery: subscribe to EEG channels
                    handler.postDelayed(() -> subscribeEeg(gatt, svc), 500);
                }
            }

            @Override
            public void onCharacteristicChanged(BluetoothGatt gatt, BluetoothGattCharacteristic ch, byte[] value) {
                sendEegData(ch.getUuid(), value);
            }

            @Deprecated
            @Override
            public void onCharacteristicChanged(BluetoothGatt gatt, BluetoothGattCharacteristic ch) {
                sendEegData(ch.getUuid(), ch.getValue());
            }

            @Override
            public void onDescriptorWrite(BluetoothGatt gatt, BluetoothGattDescriptor desc, int status) {
                if (status == BluetoothGatt.GATT_SUCCESS) {
                    subscribedChannels++;
                    Log.d(TAG, "Subscribed channel " + subscribedChannels);
                } else {
                    Log.e(TAG, "Descriptor write failed: " + status);
                }
                // Trigger next channel subscription (subscribeNextChannel handles completion)
                pendingDescriptorWrites--;
                // Find which EEG char this descriptor belongs to and subscribe next
                BluetoothGattService svc = gatt.getService(MUSE_SERVICE);
                if (svc != null) {
                    List<BluetoothGattCharacteristic> eegChars = new ArrayList<>();
                    for (UUID u : EEG_CHARS) {
                        BluetoothGattCharacteristic c = svc.getCharacteristic(u);
                        if (c != null) eegChars.add(c);
                    }
                    int nextIdx = subscribedChannels + (subscribedChannels < eegChars.size() ? 0 : 0);
                    if (nextIdx < eegChars.size()) {
                        handler.postDelayed(() -> subscribeNextChannel(gatt, eegChars, nextIdx), 250);
                    } else {
                        // All channels processed
                        if (subscribedChannels > 0) {
                            isStreaming = true;
                            JSObject r = new JSObject();
                            r.put("connected", true);
                            r.put("channels", subscribedChannels);
                            resolveConnect(r);
                        } else {
                            rejectConnect("No EEG channels subscribed");
                        }
                    }
                }
            }
        };

        try {
            // Direct connect (autoConnect=false). If status 8 timeout, retry in callback.
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                bluetoothGatt = device.connectGatt(
                    getContext(), false, gattCb,
                    BluetoothDevice.TRANSPORT_LE
                );
            } else {
                bluetoothGatt = device.connectGatt(getContext(), false, gattCb, BluetoothDevice.TRANSPORT_LE);
            }
        } catch (SecurityException e) {
            call.reject("Connect permission denied");
            return;
        }

        timeoutRunnable = () -> {
            rejectConnect("Connection timed out. Turn Muse off/on and try again.");
            if (bluetoothGatt != null) {
                try { bluetoothGatt.disconnect(); } catch (Exception ignored) {}
                try { bluetoothGatt.close(); } catch (Exception ignored) {}
                bluetoothGatt = null;
            }
        };
        handler.postDelayed(timeoutRunnable, 60000); // 60s total (includes scan + connect + retry + discovery)
    }

    @PluginMethod
    public void disconnect(PluginCall call) {
        isStreaming = false;
        if (bluetoothGatt != null) {
            try {
                BluetoothGattService svc = bluetoothGatt.getService(MUSE_SERVICE);
                if (svc != null) {
                    BluetoothGattCharacteristic ctrl = svc.getCharacteristic(CONTROL_CHAR);
                    if (ctrl != null) writeChar(bluetoothGatt, ctrl, CMD_STOP);
                }
            } catch (Exception ignored) {}
            handler.postDelayed(() -> {
                try { bluetoothGatt.disconnect(); } catch (Exception ignored) {}
                try { bluetoothGatt.close(); } catch (Exception ignored) {}
                bluetoothGatt = null;
            }, 300);
        }
        call.resolve();
    }

    @PluginMethod
    public void isConnected(PluginCall call) {
        JSObject r = new JSObject();
        r.put("connected", isStreaming);
        call.resolve(r);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private BluetoothAdapter getAdapter() {
        BluetoothManager mgr = (BluetoothManager) getContext().getSystemService(Context.BLUETOOTH_SERVICE);
        return mgr != null ? mgr.getAdapter() : null;
    }

    private boolean hasBlePermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            return ActivityCompat.checkSelfPermission(getContext(), Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED
                && ActivityCompat.checkSelfPermission(getContext(), Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED;
        }
        return ActivityCompat.checkSelfPermission(getContext(), Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED;
    }

    @PermissionCallback
    private void permCallback(PluginCall call) {
        if (hasBlePermissions()) {
            String m = call.getMethodName();
            if ("scan".equals(m)) scan(call);
            else if ("connect".equals(m)) connect(call);
        } else {
            call.reject("Bluetooth permissions required");
        }
    }

    /**
     * Refresh GATT cache using hidden API.
     * Fixes "characteristic not found" on Android 16 (Pixel 10 XL).
     */
    private void refreshGattCache(BluetoothGatt gatt) {
        try {
            Method m = gatt.getClass().getMethod("refresh");
            m.invoke(gatt);
            Log.d(TAG, "GATT cache refreshed");
        } catch (Exception e) {
            Log.w(TAG, "GATT refresh failed: " + e.getMessage());
        }
    }

    private void writeChar(BluetoothGatt gatt, BluetoothGattCharacteristic ch, byte[] value) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                gatt.writeCharacteristic(ch, value, BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE);
            } else {
                ch.setWriteType(BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE);
                ch.setValue(value);
                gatt.writeCharacteristic(ch);
            }
        } catch (SecurityException e) {
            Log.e(TAG, "Write denied: " + e.getMessage());
        }
    }

    private void subscribeEeg(BluetoothGatt gatt, BluetoothGattService svc) {
        subscribedChannels = 0;
        pendingDescriptorWrites = 0;

        // Auto-detect Muse 2 vs Muse S based on characteristic UUIDs
        List<BluetoothGattCharacteristic> allChars = svc.getCharacteristics();
        Log.d(TAG, "Service has " + allChars.size() + " characteristics");

        // Try Muse S UUIDs first (273e0013-0016)
        List<BluetoothGattCharacteristic> eegChars = new ArrayList<>();
        for (UUID u : EEG_CHARS_MUSE_S) {
            BluetoothGattCharacteristic ch = svc.getCharacteristic(u);
            if (ch != null) eegChars.add(ch);
        }
        if (!eegChars.isEmpty()) {
            EEG_CHARS = EEG_CHARS_MUSE_S;
            Log.d(TAG, "Detected Muse S EEG UUIDs (0013-0016), found " + eegChars.size() + " channels");
        }

        // Try Muse 2 UUIDs (273e0003-0006)
        if (eegChars.isEmpty()) {
            for (UUID u : EEG_CHARS_MUSE2) {
                BluetoothGattCharacteristic ch = svc.getCharacteristic(u);
                if (ch != null) eegChars.add(ch);
            }
            if (!eegChars.isEmpty()) {
                EEG_CHARS = EEG_CHARS_MUSE2;
                Log.d(TAG, "Detected Muse 2 EEG UUIDs (0003-0006), found " + eegChars.size() + " channels");
            }
        }

        if (eegChars.isEmpty()) {
            StringBuilder charList = new StringBuilder();
            for (BluetoothGattCharacteristic c : allChars) {
                charList.append(c.getUuid().toString()).append(", ");
            }
            rejectConnect("No EEG chars found (" + allChars.size() + " total: " + charList + ")");
            return;
        }

        // Subscribe one at a time with delays (BLE requires sequential descriptor writes)
        pendingDescriptorWrites = eegChars.size();
        subscribeNextChannel(gatt, eegChars, 0);
    }

    private void subscribeNextChannel(BluetoothGatt gatt, List<BluetoothGattCharacteristic> chars, int index) {
        if (index >= chars.size()) {
            // All done
            if (subscribedChannels > 0) {
                isStreaming = true;
                JSObject r = new JSObject();
                r.put("connected", true);
                r.put("channels", subscribedChannels);
                resolveConnect(r);
            } else {
                rejectConnect("Failed to subscribe to any EEG channel");
            }
            return;
        }

        BluetoothGattCharacteristic ch = chars.get(index);
        try {
            gatt.setCharacteristicNotification(ch, true);
            BluetoothGattDescriptor desc = ch.getDescriptor(CCC_DESCRIPTOR);
            if (desc != null) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    gatt.writeDescriptor(desc, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
                } else {
                    desc.setValue(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
                    gatt.writeDescriptor(desc);
                }
                // onDescriptorWrite callback will trigger next channel
            } else {
                // No descriptor needed — count as subscribed
                subscribedChannels++;
                // Subscribe next after short delay
                handler.postDelayed(() -> subscribeNextChannel(gatt, chars, index + 1), 100);
            }
        } catch (SecurityException e) {
            Log.e(TAG, "Subscribe denied: " + e.getMessage());
            handler.postDelayed(() -> subscribeNextChannel(gatt, chars, index + 1), 100);
        }
    }

    private void sendEegData(UUID charUuid, byte[] value) {
        if (value == null) return;
        int ch = -1;
        for (int i = 0; i < EEG_CHARS.length; i++) {
            if (EEG_CHARS[i].equals(charUuid)) { ch = i; break; }
        }
        if (ch < 0) return;

        JSObject data = new JSObject();
        data.put("channel", ch);
        data.put("data", Base64.encodeToString(value, Base64.NO_WRAP));
        notifyListeners("museEegData", data);
    }

    private synchronized void resolveConnect(JSObject result) {
        if (timeoutRunnable != null) {
            handler.removeCallbacks(timeoutRunnable);
            timeoutRunnable = null;
        }
        if (connectCall != null) {
            connectCall.resolve(result);
            connectCall = null;
        }
    }

    private synchronized void rejectConnect(String msg) {
        if (connectCall != null) {
            connectCall.reject(msg);
            connectCall = null;
        }
    }
}
