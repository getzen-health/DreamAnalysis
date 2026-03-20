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
    private static final UUID[] EEG_CHARS = {
        UUID.fromString("273e0003-4c4d-454d-96be-f03bac821358"),
        UUID.fromString("273e0004-4c4d-454d-96be-f03bac821358"),
        UUID.fromString("273e0005-4c4d-454d-96be-f03bac821358"),
        UUID.fromString("273e0006-4c4d-454d-96be-f03bac821358"),
    };
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

        ScanFilter filter = new ScanFilter.Builder()
            .setServiceUuid(new ParcelUuid(MUSE_SERVICE))
            .build();
        ScanSettings settings = new ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build();

        try {
            scanner.startScan(Collections.singletonList(filter), settings, scanCb);
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
        }, 8000);
    }

    @PluginMethod
    public void connect(PluginCall call) {
        String deviceId = call.getString("deviceId");
        if (deviceId == null) { call.reject("deviceId required"); return; }
        if (!hasBlePermissions()) { requestAllPermissions(call, "permCallback"); return; }

        BluetoothAdapter adapter = getAdapter();
        if (adapter == null) { call.reject("Bluetooth not available"); return; }

        // Cleanup existing connection
        if (bluetoothGatt != null) {
            try { bluetoothGatt.disconnect(); } catch (Exception ignored) {}
            try { bluetoothGatt.close(); } catch (Exception ignored) {}
            bluetoothGatt = null;
        }
        isStreaming = false;
        subscribedChannels = 0;

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
                if (newState == BluetoothProfile.STATE_CONNECTED) {
                    // CRITICAL: refresh GATT cache to fix Android 16 "characteristic not found"
                    refreshGattCache(gatt);
                    handler.postDelayed(() -> {
                        try {
                            gatt.discoverServices();
                        } catch (SecurityException e) {
                            rejectConnect("Permission denied: " + e.getMessage());
                        }
                    }, 1500);
                } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                    isStreaming = false;
                    notifyListeners("museDisconnected", new JSObject());
                    rejectConnect("Disconnected (status: " + status + ")");
                }
            }

            @Override
            public void onServicesDiscovered(BluetoothGatt gatt, int status) {
                Log.d(TAG, "Services discovered: " + status + " count: " + gatt.getServices().size());
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

                handler.postDelayed(() -> {
                    BluetoothGattCharacteristic ctrl = svc.getCharacteristic(CONTROL_CHAR);
                    if (ctrl == null) {
                        rejectConnect("Control characteristic not found");
                        return;
                    }
                    writeChar(gatt, ctrl, CMD_PRESET);
                    handler.postDelayed(() -> {
                        writeChar(gatt, ctrl, CMD_START);
                        handler.postDelayed(() -> subscribeEeg(gatt, svc), 500);
                    }, 500);
                }, 500);
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
                    Log.d(TAG, "Subscribed " + subscribedChannels + "/4");
                }
                pendingDescriptorWrites--;
                if (pendingDescriptorWrites <= 0) {
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
        };

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                bluetoothGatt = device.connectGatt(getContext(), false, gattCb, BluetoothDevice.TRANSPORT_LE, BluetoothDevice.PHY_LE_1M);
            } else {
                bluetoothGatt = device.connectGatt(getContext(), false, gattCb, BluetoothDevice.TRANSPORT_LE);
            }
        } catch (SecurityException e) {
            call.reject("Connect permission denied");
            return;
        }

        handler.postDelayed(() -> rejectConnect("Connection timed out"), 30000);
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

        for (int i = 0; i < EEG_CHARS.length; i++) {
            BluetoothGattCharacteristic ch = svc.getCharacteristic(EEG_CHARS[i]);
            if (ch == null) {
                Log.e(TAG, "EEG char " + i + " not found");
                continue;
            }
            try {
                gatt.setCharacteristicNotification(ch, true);
                BluetoothGattDescriptor desc = ch.getDescriptor(CCC_DESCRIPTOR);
                if (desc != null) {
                    pendingDescriptorWrites++;
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                        gatt.writeDescriptor(desc, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
                    } else {
                        desc.setValue(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
                        gatt.writeDescriptor(desc);
                    }
                }
            } catch (SecurityException e) {
                Log.e(TAG, "Subscribe denied ch " + i + ": " + e.getMessage());
            }

            // Wait between descriptor writes (BLE requires sequential)
            try { Thread.sleep(250); } catch (InterruptedException ignored) {}
        }

        // If no descriptor writes were needed, resolve immediately
        if (pendingDescriptorWrites == 0) {
            if (subscribedChannels > 0) {
                isStreaming = true;
                JSObject r = new JSObject();
                r.put("connected", true);
                r.put("channels", subscribedChannels);
                resolveConnect(r);
            } else {
                rejectConnect("No EEG channels available");
            }
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
