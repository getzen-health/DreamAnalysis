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

    // Commands matching muse-js protocol: halt → preset → stream → resume
    private static final byte[] CMD_HALT    = {0x02, 0x68, 0x0a};           // "h\n"
    private static final byte[] CMD_PRESET  = {0x04, 0x70, 0x32, 0x31, 0x0a}; // "p21\n"
    private static final byte[] CMD_STREAM  = {0x02, 0x73, 0x0a};           // "s\n"
    private static final byte[] CMD_RESUME  = {0x02, 0x64, 0x0a};           // "d\n"
    private static final byte[] CMD_STOP    = CMD_HALT;

    // Athena (Muse S) specific commands
    private static final byte[] CMD_VERSION6    = {0x03, 0x76, 0x36, 0x0a};                     // "v6\n"
    private static final byte[] CMD_STATUS      = CMD_STREAM;                                     // "s\n" (same byte sequence)
    private static final byte[] CMD_DC001       = {0x06, 0x64, 0x63, 0x30, 0x30, 0x31, 0x0a};   // "dc001\n"
    private static final byte[] CMD_LOW_LATENCY = {0x03, 0x4c, 0x31, 0x0a};                     // "L1\n"

    // Athena data characteristic (multiplexed — all 4 EEG channels in one notification)
    private static final UUID ATHENA_DATA_CHAR = UUID.fromString("273e0013-4c4d-454d-96be-f03bac821358");
    private static final UUID ATHENA_AUX_CHAR  = UUID.fromString("273e0014-4c4d-454d-96be-f03bac821358");

    // Athena packet constants
    private static final int ATHENA_HEADER_SIZE  = 14;
    private static final int ATHENA_TAG_EEG_4CH  = 0x11;
    private static final int ATHENA_EEG_4CH_SIZE = 28;
    private static final double ATHENA_UV_SCALE  = 1450.0 / 16383.0;

    private final Handler handler = new Handler(Looper.getMainLooper());
    private BluetoothGatt bluetoothGatt;
    private PluginCall connectCall;
    private boolean isStreaming = false;
    private boolean isAthena = false;
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
            sendDiag("Scan done: " + devices.size() + " Muse device(s)");
            JSObject result = new JSObject();
            JSONArray arr = new JSONArray();
            for (JSONObject d : devices) arr.put(d);
            result.put("devices", arr);
            call.resolve(result);
        }, 5000);
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
        isAthena = false;
        athenaPacketCount = 0;
        subscribedChannels = 0;
        connectRetried = false;
        connectPhase = 0;
        writePending = false;
        commandsSentAfterSubscribe = false;
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
                sendDiag("BLE state=" + newState + " status=" + status);
                if (newState == BluetoothProfile.STATE_CONNECTED) {
                    sendDiag("Connected. Refreshing GATT...");
                    refreshGattCache(gatt);
                    handler.postDelayed(() -> {
                        try {
                            sendDiag("Discovering services...");
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

                int charCount = svc.getCharacteristics().size();
                sendDiag("Phase " + connectPhase + ": " + charCount + " chars found");
                for (BluetoothGattCharacteristic c : svc.getCharacteristics()) {
                    String shortUuid = c.getUuid().toString().substring(4, 8);
                    sendDiag("  " + shortUuid + " props=0x" + Integer.toHexString(c.getProperties()));
                }

                if (connectPhase == 0) {
                    // Phase 0: First discovery.
                    // Step 1: Subscribe to control char (Muse requires this before processing commands)
                    connectPhase = 1;
                    BluetoothGattCharacteristic ctrl = svc.getCharacteristic(CONTROL_CHAR);
                    if (ctrl == null) {
                        rejectConnect("Control characteristic not found");
                        return;
                    }

                    sendDiag("Subscribing control char...");
                    // Enable control char notifications
                    try {
                        gatt.setCharacteristicNotification(ctrl, true);
                        BluetoothGattDescriptor desc = ctrl.getDescriptor(CCC_DESCRIPTOR);
                        if (desc != null) {
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                                gatt.writeDescriptor(desc, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
                            } else {
                                desc.setValue(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
                                gatt.writeDescriptor(desc);
                            }
                            Log.d(TAG, "Control char notification descriptor written");
                            // onDescriptorWrite will continue the flow
                            return;
                        }
                    } catch (SecurityException e) {
                        Log.w(TAG, "Control char subscribe failed: " + e.getMessage());
                    }
                    // No descriptor or subscribe failed — continue with commands anyway
                    sendMuseCommands(gatt, ctrl);
                } else {
                    // Phase 2: Re-discovery completed. Subscribe to EEG.
                    Log.d(TAG, "Phase 2: Re-discovery done (" + svc.getCharacteristics().size() + " chars). Subscribing...");
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
                UUID charUuid = desc.getCharacteristic().getUuid();
                Log.d(TAG, "onDescriptorWrite: char=" + charUuid + " status=" + status + " phase=" + connectPhase);

                // Phase 1: control char subscribed → subscribe EEG chars BEFORE commands
                if (connectPhase == 1 && charUuid.equals(CONTROL_CHAR)) {
                    sendDiag("Control subscribed. Now subscribing EEG BEFORE commands...");
                    // Detect Athena
                    BluetoothGattService svc = gatt.getService(MUSE_SERVICE);
                    boolean has0013 = svc != null && svc.getCharacteristic(ATHENA_DATA_CHAR) != null;
                    boolean hasMuse2 = false;
                    if (svc != null) {
                        for (UUID u : EEG_CHARS_MUSE2) {
                            if (svc.getCharacteristic(u) != null) { hasMuse2 = true; break; }
                        }
                    }
                    isAthena = has0013 && !hasMuse2;
                    connectPhase = 2; // EEG subscribe phase
                    subscribedChannels = 0;
                    List<BluetoothGattCharacteristic> eegChars = findEegChars(svc);
                    sendDiag((isAthena ? "Athena" : "Muse2") + ": subscribing " + eegChars.size() + " EEG chars...");
                    if (!eegChars.isEmpty()) {
                        subscribeNextChannel(gatt, eegChars, 0);
                    } else {
                        // No EEG chars yet — send commands then re-discover
                        sendDiag("No EEG chars in phase 1. Sending commands...");
                        BluetoothGattCharacteristic ctrl = svc != null ? svc.getCharacteristic(CONTROL_CHAR) : null;
                        if (ctrl != null) sendMuseCommands(gatt, ctrl);
                    }
                    return;
                }

                // Phase 2: EEG channel descriptor written
                if (status == BluetoothGatt.GATT_SUCCESS) {
                    subscribedChannels++;
                    Log.d(TAG, "EEG channel " + subscribedChannels + " subscribed");
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

    // ── Send muse-js command sequence: halt → preset → stream → resume ────

    private void sendMuseCommands(BluetoothGatt gatt, BluetoothGattCharacteristic ctrl) {
        writeQueue.clear();
        writePending = false;

        // Detect Athena: check if 0013 exists but 0003-0006 do NOT
        BluetoothGattService svc = gatt.getService(MUSE_SERVICE);
        boolean has0013 = svc != null && svc.getCharacteristic(ATHENA_DATA_CHAR) != null;
        boolean hasMuse2Chars = false;
        if (svc != null) {
            for (UUID u : EEG_CHARS_MUSE2) {
                if (svc.getCharacteristic(u) != null) { hasMuse2Chars = true; break; }
            }
        }
        isAthena = has0013 && !hasMuse2Chars;

        if (isAthena) {
            sendDiag("Athena detected. Sending: v6→s→h→p21→s→dc001→dc001→L1→s");
            enqueueWrite(gatt, ctrl, CMD_VERSION6, "v6");
            enqueueWrite(gatt, ctrl, CMD_STATUS, "status1");
            enqueueWrite(gatt, ctrl, CMD_HALT, "halt");
            enqueueWrite(gatt, ctrl, CMD_PRESET, "preset");
            enqueueWrite(gatt, ctrl, CMD_STATUS, "status2");
            enqueueWrite(gatt, ctrl, CMD_DC001, "dc001-1");
            enqueueWrite(gatt, ctrl, CMD_DC001, "dc001-2");  // sent TWICE
            enqueueWrite(gatt, ctrl, CMD_LOW_LATENCY, "L1");
            enqueueWrite(gatt, ctrl, CMD_STATUS, "status3");
        } else {
            sendDiag("Muse 2 detected. Sending: halt→preset→stream→resume");
            enqueueWrite(gatt, ctrl, CMD_HALT, "halt");
            enqueueWrite(gatt, ctrl, CMD_PRESET, "preset");
            enqueueWrite(gatt, ctrl, CMD_STREAM, "stream");
            enqueueWrite(gatt, ctrl, CMD_RESUME, "resume");
        }

        writeQueue.add(() -> {
            sendDiag("Commands done. Waiting 3s for reconfigure...");
            handler.postDelayed(() -> trySubscribeOrRediscover(gatt), 3000);
        });
        processWriteQueue();
    }

    // ── Core: try subscribe, fall back to re-discovery ──────────────────────

    private void trySubscribeOrRediscover(BluetoothGatt gatt) {
        BluetoothGattService svc = gatt.getService(MUSE_SERVICE);
        if (svc == null) {
            rejectConnect("Muse service lost after commands");
            return;
        }

        List<BluetoothGattCharacteristic> eegChars = findEegChars(svc);
        sendDiag("EEG chars in cache: " + eegChars.size());
        if (!eegChars.isEmpty()) {
            sendDiag("Subscribing to " + eegChars.size() + " EEG channels...");
            subscribeEeg(gatt, svc);
            return;
        }

        sendDiag("No EEG chars. Re-discovering (phase 2)...");
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
            // Muse S control char only supports writeWithoutResponse.
            // Always use NO_RESPONSE and proceed with delay (no callback expected).
            Log.d(TAG, "Writing " + label + " (" + value.length + " bytes) NO_RESPONSE");
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    gatt.writeCharacteristic(ch, value, BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE);
                } else {
                    ch.setWriteType(BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE);
                    ch.setValue(value);
                    gatt.writeCharacteristic(ch);
                }
            } catch (SecurityException e) {
                Log.e(TAG, "Write " + label + " denied: " + e.getMessage());
            }
            // No callback for NO_RESPONSE — proceed after 200ms
            writePending = false;
            handler.postDelayed(this::processWriteQueue, 200);
        });
    }

    private void processWriteQueue() {
        if (writePending || writeQueue.isEmpty()) return;
        Runnable next = writeQueue.poll();
        if (next != null) next.run();
    }

    // ── Subscribe helpers ───────────────────────────────────────────────────

    private List<BluetoothGattCharacteristic> findEegChars(BluetoothGattService svc) {
        List<BluetoothGattCharacteristic> result = new ArrayList<>();

        // Athena: single multiplexed data characteristic (0013)
        if (isAthena) {
            BluetoothGattCharacteristic dataChar = svc.getCharacteristic(ATHENA_DATA_CHAR);
            if (dataChar != null) {
                result.add(dataChar);
                // Also add aux char (0014) for indicate subscription
                BluetoothGattCharacteristic auxChar = svc.getCharacteristic(ATHENA_AUX_CHAR);
                if (auxChar != null) result.add(auxChar);
                EEG_CHARS = EEG_CHARS_MUSE_S;
                sendDiag("Athena: found data char 0013" + (auxChar != null ? " + aux 0014" : ""));
            }
            return result;
        }

        // Try Muse S first (non-Athena with per-channel chars), then Muse 2
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
        String deviceType = isAthena ? "Athena" : (EEG_CHARS == EEG_CHARS_MUSE_S ? "Muse S" : "Muse 2");
        Log.d(TAG, "subscribeEeg: found " + eegChars.size() + " chars (type: " + deviceType + ")");

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
                // Athena: 0014 uses INDICATE (CCCD 0x0002), 0013 and others use NOTIFY (CCCD 0x0001)
                byte[] cccdValue;
                if (isAthena && ch.getUuid().equals(ATHENA_AUX_CHAR)) {
                    cccdValue = BluetoothGattDescriptor.ENABLE_INDICATION_VALUE;
                    sendDiag("Athena: subscribing 0014 with INDICATE");
                } else {
                    cccdValue = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE;
                }

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    gatt.writeDescriptor(desc, cccdValue);
                } else {
                    desc.setValue(cccdValue);
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

    private boolean commandsSentAfterSubscribe = false;

    private void finishSubscription() {
        if (subscribedChannels <= 0) {
            rejectConnect("Failed to subscribe to any EEG channel");
            return;
        }

        int logicalChannels = isAthena ? 4 : subscribedChannels;
        sendDiag(logicalChannels + " EEG channels subscribed" + (isAthena ? " (Athena)" : ""));

        // If commands haven't been sent yet (subscribe-first flow), send them NOW
        if (!commandsSentAfterSubscribe) {
            commandsSentAfterSubscribe = true;
            sendDiag("EEG subscribed. NOW sending start commands...");
            BluetoothGattService svc = bluetoothGatt.getService(MUSE_SERVICE);
            BluetoothGattCharacteristic ctrl = svc != null ? svc.getCharacteristic(CONTROL_CHAR) : null;
            if (ctrl != null) {
                // Send commands, then resolve on completion
                sendMuseCommandsThenResolve(logicalChannels);
            } else {
                rejectConnect("Control char not found after subscribe");
            }
            return;
        }

        // Commands already sent — we're done
        isStreaming = true;
        sendDiag("SUCCESS: Streaming " + logicalChannels + " channels!");
        JSObject r = new JSObject();
        r.put("connected", true);
        r.put("channels", logicalChannels);
        resolveConnect(r);
    }

    /** Send Athena/Muse2 commands after EEG chars are subscribed, then resolve. */
    private void sendMuseCommandsThenResolve(int logicalChannels) {
        BluetoothGattService svc = bluetoothGatt.getService(MUSE_SERVICE);
        BluetoothGattCharacteristic ctrl = svc != null ? svc.getCharacteristic(CONTROL_CHAR) : null;
        if (ctrl == null) { rejectConnect("Control char lost"); return; }

        writeQueue.clear();
        writePending = false;

        if (isAthena) {
            sendDiag("Sending Athena: v6→s→h→p21→s→dc001→dc001→L1→s");
            enqueueWrite(bluetoothGatt, ctrl, CMD_VERSION6, "v6");
            enqueueWrite(bluetoothGatt, ctrl, CMD_STATUS, "status1");
            enqueueWrite(bluetoothGatt, ctrl, CMD_HALT, "halt");
            enqueueWrite(bluetoothGatt, ctrl, CMD_PRESET, "preset");
            enqueueWrite(bluetoothGatt, ctrl, CMD_STATUS, "status2");
            enqueueWrite(bluetoothGatt, ctrl, CMD_DC001, "dc001-1");
            enqueueWrite(bluetoothGatt, ctrl, CMD_DC001, "dc001-2");
            enqueueWrite(bluetoothGatt, ctrl, CMD_LOW_LATENCY, "L1");
            enqueueWrite(bluetoothGatt, ctrl, CMD_STATUS, "status3");
        } else {
            sendDiag("Sending Muse2: h→p21→s→d");
            enqueueWrite(bluetoothGatt, ctrl, CMD_HALT, "halt");
            enqueueWrite(bluetoothGatt, ctrl, CMD_PRESET, "preset");
            enqueueWrite(bluetoothGatt, ctrl, CMD_STREAM, "stream");
            enqueueWrite(bluetoothGatt, ctrl, CMD_RESUME, "resume");
        }

        writeQueue.add(() -> {
            sendDiag("Commands sent! Streaming should start...");
            isStreaming = true;
            JSObject r = new JSObject();
            r.put("connected", true);
            r.put("channels", logicalChannels);
            resolveConnect(r);
        });
        processWriteQueue();
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

    private int athenaPacketCount = 0;

    private void sendEegData(UUID charUuid, byte[] value) {
        if (value == null) return;

        // Athena: multiplexed data on 0013 — parse and emit per-channel events
        if (isAthena && charUuid.equals(ATHENA_DATA_CHAR)) {
            parseAndSendAthenaData(value);
            return;
        }

        // Standard Muse 2 / Muse S per-channel path
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

    /**
     * Parse Athena multiplexed data notification and emit as separate channel events.
     *
     * Packet: [14-byte header][subpacket1][subpacket2]...
     * Tag 0x11: 4-channel EEG, 28 bytes (4 samples x 4 channels x 14-bit packed as uint16 BE)
     *
     * We synthesize 20-byte Muse 2-compatible packets per channel so the JS side
     * can use the same onEegNotification decoder.
     *
     * Actually, since JS side now has onAthenaDataNotification, we send the raw
     * Athena packet as channel=-1 to signal Athena format. The JS side routes
     * based on channel value. But it's simpler to just forward the raw bytes
     * and let JS parse — the JS _connectNativeMusePlugin listener already
     * routes to onEegNotification. We need to add Athena routing there.
     *
     * Simplest approach: send raw bytes with channel=-1, JS side detects Athena.
     */
    private int athenaChannelRotation = 0;

    /**
     * Athena data on 0013: with p21 preset, sends Muse 2-compatible 20-byte packets
     * multiplexed across 4 channels in round-robin order.
     * Just forward each packet as a channel event using rotation.
     * Also log the first few packets' raw bytes to understand format.
     */
    private void parseAndSendAthenaData(byte[] value) {
        athenaPacketCount++;

        if (athenaPacketCount <= 5) {
            StringBuilder hex = new StringBuilder();
            for (int i = 0; i < Math.min(value.length, 24); i++) {
                hex.append(String.format("%02x ", value[i] & 0xFF));
            }
            sendDiag("Athena pkt#" + athenaPacketCount + " len=" + value.length + ": " + hex.toString().trim());
        }

        if (value.length < 20) return;

        // Tag at byte 9 — only process EEG (0x11)
        int tag = value[9] & 0xFF;
        if (tag != 0x11) return;

        // Athena EEG: byte[1]=seq, byte[9]=tag(0x11), bytes[10-11]=sub-index
        // Bytes 12-19 = 8 bytes of EEG data
        // Channel from sequence: 4 EEG packets per cycle, one per channel
        int seq = value[1] & 0xFF;
        int ch = seq % 4;

        // Decode EEG: try 12-bit packed pairs from bytes 12-19 (same as Muse 2 packing)
        // Each 3 bytes → 2 twelve-bit samples: s0 = (b0<<4)|(b1>>4), s1 = ((b1&0xf)<<8)|b2
        // 8 bytes → 5 twelve-bit samples (last byte only contributes to 1 sample)
        int dataStart = 12;
        int dataLen = Math.min(value.length - dataStart, 9); // up to 9 bytes = 6 samples
        int numPairs = dataLen / 3;
        double[] samples = new double[numPairs * 2];
        int si = 0;
        for (int p = 0; p < numPairs; p++) {
            int off = dataStart + p * 3;
            if (off + 2 >= value.length) break;
            int b0 = value[off] & 0xFF;
            int b1 = value[off + 1] & 0xFF;
            int b2 = value[off + 2] & 0xFF;
            int s0 = ((b0 << 4) | (b1 >> 4)) & 0xFFF;
            int s1 = (((b1 & 0xF) << 8) | b2) & 0xFFF;
            // 12-bit centered at 2048, scale to microvolts (same as Muse 2)
            samples[si++] = (s0 - 2048) * 0.48828125;
            samples[si++] = (s1 - 2048) * 0.48828125;
        }

        // Send decoded samples as JSON (not raw bytes)
        JSObject data = new JSObject();
        data.put("channel", ch);
        data.put("athena", true);
        StringBuilder samplesStr = new StringBuilder();
        for (int i = 0; i < si; i++) {
            if (i > 0) samplesStr.append(",");
            samplesStr.append(String.format("%.2f", samples[i]));
        }
        data.put("samples", samplesStr.toString());
        notifyListeners("museEegData", data);

        if (athenaPacketCount == 50 || athenaPacketCount == 200) {
            sendDiag("Athena: " + athenaPacketCount + " pkts ch=" + ch +
                " s0=" + String.format("%.1f", samples[0]) + "uV");
        }
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
        sendDiag("ERROR: " + msg);
        if (connectCall != null) {
            connectCall.reject(msg);
            connectCall = null;
        }
    }

    /** Send a diagnostic message to the JS side for on-screen display */
    private void sendDiag(String msg) {
        Log.d(TAG, "DIAG: " + msg);
        JSObject data = new JSObject();
        data.put("message", msg);
        notifyListeners("museDiag", data);
    }
}
