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
 * Native Android BLE plugin for Muse headbands (Muse 2 + Muse S).
 *
 * Connection flow (matches Web Bluetooth approach):
 * 1. Connect GATT + discover services
 * 2. Write preset command (wait for onCharacteristicWrite callback)
 * 3. Write start command (wait for onCharacteristicWrite callback)
 * 4. Wait 3s for Muse to reconfigure
 * 5. Try subscribing to EEG chars from current cache
 * 6. If no EEG chars found → re-discover services, then subscribe
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
    private static final UUID[] EEG_CHARS_MUSE2 = {
        UUID.fromString("273e0003-4c4d-454d-96be-f03bac821358"),
        UUID.fromString("273e0004-4c4d-454d-96be-f03bac821358"),
        UUID.fromString("273e0005-4c4d-454d-96be-f03bac821358"),
        UUID.fromString("273e0006-4c4d-454d-96be-f03bac821358"),
    };
    private static final UUID[] EEG_CHARS_MUSE_S = {
        UUID.fromString("273e0013-4c4d-454d-96be-f03bac821358"),
        UUID.fromString("273e0014-4c4d-454d-96be-f03bac821358"),
        UUID.fromString("273e0015-4c4d-454d-96be-f03bac821358"),
        UUID.fromString("273e0016-4c4d-454d-96be-f03bac821358"),
    };
    private UUID[] EEG_CHARS = EEG_CHARS_MUSE2;
    private static final UUID CCC_DESCRIPTOR = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb");

    private static final byte[] CMD_PRESET = {0x04, 0x70, 0x32, 0x31, 0x0a};
    private static final byte[] CMD_START  = {0x02, 0x64, 0x0a};
    private static final byte[] CMD_STOP   = {0x02, 0x68, 0x0a};

    private final Handler handler = new Handler(Looper.getMainLooper());
    private BluetoothGatt bluetoothGatt;
    private PluginCall connectCall;
    private boolean isStreaming = false;
    private int subscribedChannels = 0;
    private boolean connectRetried = false;
    private Runnable timeoutRunnable = null;
    private BluetoothDevice scannedMuseDevice = null;

    // Write queue: Android BLE only allows one GATT operation at a time.
    // Queue writes and process them sequentially via onCharacteristicWrite.
    private final Queue<Runnable> writeQueue = new LinkedList<>();
    private boolean writePending = false;
    // Track which phase we're in: 0=initial, 1=commands sent, 2=rediscovery
    private int connectPhase = 0;

    // ── Scan ─────────────────────────────────────────────────────────────────

    @PluginMethod
    public void scan(PluginCall call) {
        BluetoothAdapter adapter = getAdapter();
        if (adapter == null) { call.reject("Bluetooth not available"); return; }
        if (!hasBlePermissions()) { requestAllPermissions(call, "permCallback"); return; }

        BluetoothLeScanner scanner = adapter.getBluetoothLeScanner();
        if (scanner == null) { call.reject("BLE scanner not available"); return; }

        scannedMuseDevice = null;
        List<JSONObject> devices = new ArrayList<>();

        ScanCallback scanCb = new ScanCallback() {
            @Override
            public void onScanResult(int callbackType, ScanResult result) {
                try {
                    String name = result.getDevice().getName();
                    if (name == null && result.getScanRecord() != null) name = result.getScanRecord().getDeviceName();
                    if (name == null) name = "Unknown";
                    if (!name.toLowerCase().contains("muse")) return;

                    if (scannedMuseDevice == null) {
                        scannedMuseDevice = result.getDevice();
                    }

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

        try {
            scanner.startScan(null, settings, scanCb);
        } catch (SecurityException e) {
            call.reject("Scan permission denied");
            return;
        }

        handler.postDelayed(() -> {
            try { scanner.stopScan(scanCb); } catch (Exception ignored) {}
            JSObject result = new JSObject();
            JSONArray arr = new JSONArray();
            for (JSONObject d : devices) arr.put(d);
            result.put("devices", arr);
            call.resolve(result);
        }, 12000);
    }

    // ── Connect ──────────────────────────────────────────────────────────────

    @PluginMethod
    public void connect(PluginCall call) {
        String deviceId = call.getString("deviceId");
        if (deviceId == null) { call.reject("deviceId required"); return; }
        if (!hasBlePermissions()) { requestAllPermissions(call, "permCallback"); return; }

        BluetoothAdapter adapter = getAdapter();
        if (adapter == null) { call.reject("Bluetooth not available"); return; }

        if (timeoutRunnable != null) {
            handler.removeCallbacks(timeoutRunnable);
            timeoutRunnable = null;
        }
        if (bluetoothGatt != null) {
            try { bluetoothGatt.disconnect(); } catch (Exception ignored) {}
            try { bluetoothGatt.close(); } catch (Exception ignored) {}
            bluetoothGatt = null;
        }
        isStreaming = false;
        subscribedChannels = 0;
        connectRetried = false;
        connectPhase = 0;
        writePending = false;
        writeQueue.clear();
        connectCall = null;

        BluetoothDevice device;
        if (scannedMuseDevice != null && scannedMuseDevice.getAddress().equals(deviceId)) {
            device = scannedMuseDevice;
            Log.d(TAG, "Using scanned device object (address type preserved)");
        } else {
            try {
                device = adapter.getRemoteDevice(deviceId);
                Log.d(TAG, "Using getRemoteDevice (scan device not available)");
            } catch (Exception e) {
                call.reject("Invalid device: " + deviceId);
                return;
            }
        }

        connectCall = call;

        BluetoothGattCallback gattCb = new BluetoothGattCallback() {
            @Override
            public void onConnectionStateChange(BluetoothGatt gatt, int status, int newState) {
                Log.d(TAG, "onConnectionStateChange: state=" + newState + " status=" + status);
                if (newState == BluetoothProfile.STATE_CONNECTED) {
                    Log.d(TAG, "Connected! Refreshing GATT cache...");
                    refreshGattCache(gatt);
                    handler.postDelayed(() -> {
                        try {
                            Log.d(TAG, "Discovering services (phase 0)...");
                            gatt.discoverServices();
                        } catch (SecurityException e) {
                            rejectConnect("Permission denied: " + e.getMessage());
                        }
                    }, 1500);
                } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                    isStreaming = false;
                    if (status == 8 && !connectRetried) {
                        connectRetried = true;
                        Log.w(TAG, "GATT timeout (status 8), retrying...");
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
                Log.d(TAG, "onServicesDiscovered: status=" + status + " phase=" + connectPhase
                    + " services=" + gatt.getServices().size());

                if (status != BluetoothGatt.GATT_SUCCESS) {
                    rejectConnect("Service discovery failed (status " + status + ")");
                    return;
                }

                BluetoothGattService svc = gatt.getService(MUSE_SERVICE);
                if (svc == null) {
                    Log.e(TAG, "Muse service NOT found. Available services:");
                    for (BluetoothGattService s : gatt.getServices()) {
                        Log.e(TAG, "  " + s.getUuid());
                    }
                    rejectConnect("Muse service not found");
                    return;
                }

                Log.d(TAG, "Muse service has " + svc.getCharacteristics().size() + " chars:");
                for (BluetoothGattCharacteristic c : svc.getCharacteristics()) {
                    Log.d(TAG, "  " + c.getUuid() + " props=" + c.getProperties());
                }

                if (connectPhase == 0) {
                    // Phase 0: First discovery. Send preset + start commands.
                    connectPhase = 1;
                    BluetoothGattCharacteristic ctrl = svc.getCharacteristic(CONTROL_CHAR);
                    if (ctrl == null) {
                        rejectConnect("Control characteristic not found");
                        return;
                    }

                    Log.d(TAG, "Phase 1: Queuing preset + start commands...");
                    // Queue writes — executed sequentially via onCharacteristicWrite
                    enqueueWrite(gatt, ctrl, CMD_PRESET, "preset");
                    enqueueWrite(gatt, ctrl, CMD_START, "start");
                    // After both writes complete, wait then try subscribe
                    writeQueue.add(() -> {
                        Log.d(TAG, "Commands sent. Waiting 3s for Muse to reconfigure...");
                        handler.postDelayed(() -> trySubscribeOrRediscover(gatt), 3000);
                    });
                    processWriteQueue();
                } else {
                    // Phase 2: Re-discovery completed. Try subscribing again.
                    Log.d(TAG, "Phase 2: Re-discovery done. Subscribing...");
                    subscribeEeg(gatt, svc);
                }
            }

            @Override
            public void onCharacteristicWrite(BluetoothGatt gatt, BluetoothGattCharacteristic ch, int status) {
                Log.d(TAG, "onCharacteristicWrite: " + ch.getUuid() + " status=" + status);
                writePending = false;
                processWriteQueue();
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
                    Log.e(TAG, "Descriptor write failed: status=" + status);
                }
                BluetoothGattService svc = gatt.getService(MUSE_SERVICE);
                if (svc != null) {
                    List<BluetoothGattCharacteristic> eegChars = findEegChars(svc);
                    int nextIdx = subscribedChannels;
                    if (nextIdx < eegChars.size()) {
                        handler.postDelayed(() -> subscribeNextChannel(gatt, eegChars, nextIdx), 250);
                    } else {
                        finishSubscription();
                    }
                }
            }
        };

        try {
            bluetoothGatt = device.connectGatt(
                getContext(), false, gattCb, BluetoothDevice.TRANSPORT_LE
            );
        } catch (SecurityException e) {
            call.reject("Connect permission denied");
            return;
        }

        timeoutRunnable = () -> {
            Log.e(TAG, "Connection timed out at phase " + connectPhase);
            rejectConnect("Connection timed out (phase " + connectPhase + "). Turn Muse off/on and try again.");
            if (bluetoothGatt != null) {
                try { bluetoothGatt.disconnect(); } catch (Exception ignored) {}
                try { bluetoothGatt.close(); } catch (Exception ignored) {}
                bluetoothGatt = null;
            }
        };
        handler.postDelayed(timeoutRunnable, 60000);
    }

    // ── Core: try subscribe, fall back to re-discovery ──────────────────────

    private void trySubscribeOrRediscover(BluetoothGatt gatt) {
        BluetoothGattService svc = gatt.getService(MUSE_SERVICE);
        if (svc == null) {
            rejectConnect("Muse service lost after commands");
            return;
        }

        // Check if EEG chars are already visible (works for some Muse firmware)
        List<BluetoothGattCharacteristic> eegChars = findEegChars(svc);
        if (!eegChars.isEmpty()) {
            Log.d(TAG, "EEG chars found in cache (" + eegChars.size() + "). Subscribing directly.");
            subscribeEeg(gatt, svc);
            return;
        }

        // Not found — Muse likely added new chars after preset. Re-discover.
        Log.d(TAG, "No EEG chars in cache. Re-discovering services (phase 2)...");
        connectPhase = 2;
        try {
            gatt.discoverServices();
        } catch (SecurityException e) {
            rejectConnect("Re-discovery permission denied");
            return;
        }

        // Safety: if re-discovery doesn't complete in 12s, fail with details
        handler.postDelayed(() -> {
            if (connectPhase == 2 && !isStreaming) {
                Log.e(TAG, "Re-discovery timed out. Trying to subscribe with current cache...");
                BluetoothGattService svc2 = gatt.getService(MUSE_SERVICE);
                if (svc2 != null) {
                    List<BluetoothGattCharacteristic> chars = findEegChars(svc2);
                    if (!chars.isEmpty()) {
                        subscribeEeg(gatt, svc2);
                        return;
                    }
                    // Log what we DO have
                    Log.e(TAG, "Still no EEG chars. Available:");
                    for (BluetoothGattCharacteristic c : svc2.getCharacteristics()) {
                        Log.e(TAG, "  " + c.getUuid());
                    }
                }
                rejectConnect("EEG characteristics not found after re-discovery timeout");
            }
        }, 12000);
    }

    // ── Write queue ─────────────────────────────────────────────────────────

    private void enqueueWrite(BluetoothGatt gatt, BluetoothGattCharacteristic ch, byte[] value, String label) {
        writeQueue.add(() -> {
            Log.d(TAG, "Writing " + label + " (" + value.length + " bytes)...");
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    gatt.writeCharacteristic(ch, value, BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT);
                } else {
                    ch.setWriteType(BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT);
                    ch.setValue(value);
                    gatt.writeCharacteristic(ch);
                }
                writePending = true;
            } catch (SecurityException e) {
                Log.e(TAG, "Write " + label + " denied: " + e.getMessage());
                writePending = false;
                processWriteQueue();
            }
        });
    }

    private void processWriteQueue() {
        if (writePending || writeQueue.isEmpty()) return;
        Runnable next = writeQueue.poll();
        if (next != null) next.run();
    }

    // ── Subscribe helpers ───────────────────────────────────────────────────

    private List<BluetoothGattCharacteristic> findEegChars(BluetoothGattService svc) {
        // Try Muse S first, then Muse 2
        List<BluetoothGattCharacteristic> result = new ArrayList<>();
        for (UUID u : EEG_CHARS_MUSE_S) {
            BluetoothGattCharacteristic ch = svc.getCharacteristic(u);
            if (ch != null) result.add(ch);
        }
        if (!result.isEmpty()) {
            EEG_CHARS = EEG_CHARS_MUSE_S;
            return result;
        }
        for (UUID u : EEG_CHARS_MUSE2) {
            BluetoothGattCharacteristic ch = svc.getCharacteristic(u);
            if (ch != null) result.add(ch);
        }
        if (!result.isEmpty()) {
            EEG_CHARS = EEG_CHARS_MUSE2;
        }
        return result;
    }

    private void subscribeEeg(BluetoothGatt gatt, BluetoothGattService svc) {
        subscribedChannels = 0;
        List<BluetoothGattCharacteristic> eegChars = findEegChars(svc);
        Log.d(TAG, "subscribeEeg: found " + eegChars.size() + " EEG channels (type: " +
            (EEG_CHARS == EEG_CHARS_MUSE_S ? "Muse S" : "Muse 2") + ")");

        if (eegChars.isEmpty()) {
            StringBuilder allUuids = new StringBuilder();
            for (BluetoothGattCharacteristic c : svc.getCharacteristics()) {
                allUuids.append(c.getUuid().toString()).append(", ");
            }
            rejectConnect("No EEG chars found (" + svc.getCharacteristics().size() + " total: " + allUuids + ")");
            return;
        }

        subscribeNextChannel(gatt, eegChars, 0);
    }

    private void subscribeNextChannel(BluetoothGatt gatt, List<BluetoothGattCharacteristic> chars, int index) {
        if (index >= chars.size()) {
            finishSubscription();
            return;
        }

        BluetoothGattCharacteristic ch = chars.get(index);
        Log.d(TAG, "Subscribing to channel " + index + ": " + ch.getUuid());
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
                // onDescriptorWrite triggers next
            } else {
                Log.d(TAG, "No CCC descriptor on channel " + index + ", counting as subscribed");
                subscribedChannels++;
                handler.postDelayed(() -> subscribeNextChannel(gatt, chars, index + 1), 100);
            }
        } catch (SecurityException e) {
            Log.e(TAG, "Subscribe ch" + index + " denied: " + e.getMessage());
            handler.postDelayed(() -> subscribeNextChannel(gatt, chars, index + 1), 100);
        }
    }

    private void finishSubscription() {
        if (subscribedChannels > 0) {
            isStreaming = true;
            Log.d(TAG, "SUCCESS: Streaming " + subscribedChannels + " EEG channels!");
            JSObject r = new JSObject();
            r.put("connected", true);
            r.put("channels", subscribedChannels);
            resolveConnect(r);
        } else {
            rejectConnect("Failed to subscribe to any EEG channel");
        }
    }

    // ── Disconnect ──────────────────────────────────────────────────────────

    @PluginMethod
    public void disconnect(PluginCall call) {
        isStreaming = false;
        if (bluetoothGatt != null) {
            try {
                BluetoothGattService svc = bluetoothGatt.getService(MUSE_SERVICE);
                if (svc != null) {
                    BluetoothGattCharacteristic ctrl = svc.getCharacteristic(CONTROL_CHAR);
                    if (ctrl != null) {
                        try {
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                                bluetoothGatt.writeCharacteristic(ctrl, CMD_STOP, BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE);
                            } else {
                                ctrl.setWriteType(BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE);
                                ctrl.setValue(CMD_STOP);
                                bluetoothGatt.writeCharacteristic(ctrl);
                            }
                        } catch (SecurityException ignored) {}
                    }
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

    private void refreshGattCache(BluetoothGatt gatt) {
        try {
            Method m = gatt.getClass().getMethod("refresh");
            m.invoke(gatt);
            Log.d(TAG, "GATT cache refreshed");
        } catch (Exception e) {
            Log.w(TAG, "GATT refresh not available: " + e.getMessage());
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
        Log.e(TAG, "REJECT: " + msg);
        if (connectCall != null) {
            connectCall.reject(msg);
            connectCall = null;
        }
    }
}
