import Foundation
import Capacitor
import CloudKit

// MARK: - CloudKit Manager

private class ICloudSyncManager {
    let container = CKContainer(identifier: "iCloud.health.getzen.antaiai")
    var privateDB: CKDatabase { container.privateCloudDatabase }

    func isAvailable() async -> Bool {
        do {
            let status = try await container.accountStatus()
            return status == .available
        } catch {
            return false
        }
    }

    /// Upload the SQLite database file to iCloud private database
    func backup(dbPath: URL) async throws -> Date {
        let asset = CKAsset(fileURL: dbPath)
        let recordID = CKRecord.ID(recordName: "antaiai_db_backup")

        // Try to fetch existing record to update, or create new
        let record: CKRecord
        do {
            record = try await privateDB.record(for: recordID)
        } catch {
            record = CKRecord(recordType: "UserDatabase", recordID: recordID)
        }

        record["database"] = asset
        record["backupDate"] = Date() as CKRecordValue
        record["appVersion"] = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String as CKRecordValue?

        try await privateDB.save(record)
        return Date()
    }

    /// Download the SQLite database file from iCloud
    func restore(to destination: URL) async throws -> Date? {
        let recordID = CKRecord.ID(recordName: "antaiai_db_backup")
        do {
            let record = try await privateDB.record(for: recordID)
            guard let asset = record["database"] as? CKAsset,
                  let assetURL = asset.fileURL else { return nil }

            try FileManager.default.copyItem(at: assetURL, to: destination)
            return record["backupDate"] as? Date
        } catch {
            return nil // No backup found yet
        }
    }

    func getLastBackupDate() async throws -> Date? {
        let recordID = CKRecord.ID(recordName: "antaiai_db_backup")
        do {
            let record = try await privateDB.record(for: recordID)
            return record["backupDate"] as? Date
        } catch {
            return nil
        }
    }
}

// MARK: - Capacitor Plugin

@objc(ICloudSyncPlugin)
public class ICloudSyncPlugin: CAPPlugin {
    private let syncManager = ICloudSyncManager()

    /// Resolve the SQLite database file path from the db name
    private func dbURL(name: String) -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("\(name)SQLite.db")
    }

    @objc func isAvailable(_ call: CAPPluginCall) {
        Task {
            let available = await syncManager.isAvailable()
            call.resolve(["available": available])
        }
    }

    @objc func backup(_ call: CAPPluginCall) {
        guard let dbName = call.getString("dbName") else {
            call.reject("dbName required")
            return
        }
        let dbPath = dbURL(name: dbName)
        guard FileManager.default.fileExists(atPath: dbPath.path) else {
            call.reject("Database file not found at \(dbPath.path)")
            return
        }
        Task {
            do {
                let date = try await syncManager.backup(dbPath: dbPath)
                let formatter = ISO8601DateFormatter()
                call.resolve(["success": true, "timestamp": formatter.string(from: date)])
            } catch {
                call.reject("Backup failed: \(error.localizedDescription)")
            }
        }
    }

    @objc func restore(_ call: CAPPluginCall) {
        guard let dbName = call.getString("dbName") else {
            call.reject("dbName required")
            return
        }
        let destination = dbURL(name: dbName)
        // Backup existing DB before overwrite
        let backup = dbURL(name: "\(dbName)_before_restore")
        try? FileManager.default.copyItem(at: destination, to: backup)

        Task {
            do {
                let restoredDate = try await syncManager.restore(to: destination)
                let formatter = ISO8601DateFormatter()
                call.resolve([
                    "success": restoredDate != nil,
                    "restoredAt": restoredDate.map { formatter.string(from: $0) } as Any
                ])
            } catch {
                call.reject("Restore failed: \(error.localizedDescription)")
            }
        }
    }

    @objc func getLastBackupDate(_ call: CAPPluginCall) {
        Task {
            do {
                let date = try await syncManager.getLastBackupDate()
                let formatter = ISO8601DateFormatter()
                call.resolve(["date": date.map { formatter.string(from: $0) } as Any])
            } catch {
                call.resolve(["date": NSNull()])
            }
        }
    }
}
