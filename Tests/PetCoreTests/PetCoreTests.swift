import Darwin
import Foundation
import Testing
@testable import PetCore

@MainActor
struct PetCoreTests {
    private func event(_ json: String) throws -> PetEvent {
        try PetEvent.decode(Data(json.utf8))
    }

    @Test(arguments: [
        "{}", #"{"state":"busy"}"#, #"{"state":"idle","ttl":-1}"#,
        #"{"text":false}"#, #"{"animation":{"id":"jump","command_id":1,"repeats":0}}"#,
        #"{"motion":{"x":0,"y":0,"seconds":0,"command_id":2}}"#,
        #"{"sleeping":"true"}"#, #"{"text":"Rejected","delivery_id":"invalid"}"#
    ])
    func rejectsMalformedEvents(json: String) {
        #expect(throws: (any Error).self) { try event(json) }
    }

    @Test
    func decodesLongSpeechAndActivityState() throws {
        let speech = String(repeating: "x", count: 4096)
        #expect(try event("{\"text\":\"\(speech)\"}").text == speech)
        #expect(try event(#"{"state":"working"}"#).state == .working)
    }

    @Test
    func speechExpiresWithoutChangingState() async throws {
        let model = PetModel()
        model.apply(try event("{\"state\":\"working\",\"text\":\"Working\",\"ttl\":0.03}"))
        #expect(model.text == "Working")
        try await waitUntil { model.text == nil }
        #expect(model.state == .working)
    }

    @Test
    func newSpeechSurvivesEarlierExpiration() throws {
        var expirations: [DispatchWorkItem] = []
        let model = PetModel { _, work in expirations.append(work) }
        model.apply(try event("{\"state\":\"working\",\"text\":\"First\",\"ttl\":0.02}"))
        model.apply(try event("{\"text\":\"Second\",\"ttl\":1}"))
        try #require(expirations.count == 2)
        expirations[0].perform()
        #expect(model.text == "Second")
        #expect(model.state == .working)
    }

    @Test
    func stateChangesPreserveSpeechAndEmptyTextClearsIt() throws {
        let model = PetModel()
        model.apply(try event(#"{"state":"working","text":"Second","ttl":1}"#))
        model.apply(try event("{\"state\":\"idle\"}"))
        #expect(model.text == "Second")
        #expect(model.state == .idle)
        model.apply(try event("{\"text\":\"\"}"))
        #expect(model.text == nil)
    }

    @Test
    func speechRetriesAcknowledgeDeliveryAndPreserveCurrentSpeech() throws {
        var expirations: [DispatchWorkItem] = []
        let model = PetModel { _, work in expirations.append(work) }
        let identifier = UUID().uuidString
        let speech = try event("{\"text\":\"Accepted\",\"ttl\":0.02,\"delivery_id\":\"\(identifier)\"}")
        #expect(model.apply(speech) == identifier)
        #expect(model.text == "Accepted")
        let expiration = try #require(expirations.first)
        expiration.perform()
        #expect(model.text == nil)
        #expect(model.apply(speech) == identifier)
        #expect(model.text == nil)
        model.apply(try event("{\"text\":\"New speech\",\"ttl\":1}"))
        #expect(model.apply(speech) == identifier)
        #expect(model.text == "New speech")
    }

    @Test(arguments: [
        #"{"text":"   ","ttl":1}"#, #"{"text":"Hidden","ttl":0}"#, #"{"state":"working"}"#
    ])
    func acknowledgesOnlyVisibleSpeech(input: String) throws {
        let model = PetModel()
        let json = input.dropLast() + ",\"delivery_id\":\"\(UUID().uuidString)\"}"
        #expect(model.apply(try event(String(json))) == nil)
        #expect(model.text == nil)
    }

    @Test
    func socketReceivesValidEventsAfterInvalidJSON() async throws {
        let path = "/tmp/pet-test-\(UUID().uuidString).sock"
        var received: [PetEvent] = []
        let server = try SocketServer(path: path) { event in
            received.append(event)
        }
        defer { server.stop() }
        let client = try makeClient(path: path)
        defer { close(client) }
        try send(Data("invalid JSON".utf8), to: client)
        try send(Data("{\"state\":\"error\",\"text\":\"Quoted \\\"text\\\" 🐱\"}".utf8), to: client)
        try await waitUntil { !received.isEmpty }
        try #require(received.count == 1)
        #expect(received[0].state == .error)
        #expect(received[0].text == "Quoted \"text\" 🐱")
        var info = stat()
        #expect(lstat(path, &info) == 0)
        #expect(info.st_mode & 0o777 == 0o600)
        #expect(throws: (any Error).self) { try SocketServer(path: path) { _ in } }
    }

    @Test
    func napCommandsPreserveActivityAndWakeIndependently() throws {
        let model = PetModel()
        model.apply(try event("{\"sleeping\":true}"))
        #expect(model.sleeping && model.state == .idle)
        model.apply(try event("{\"state\":\"idle\"}"))
        #expect(model.sleeping)
        model.apply(try event("{\"sleeping\":false,\"state\":\"working\"}"))
        #expect(!model.sleeping && model.state == .working)
    }

    @Test
    func socketReceivesLongSpeechWithoutTruncation() async throws {
        let path = "/tmp/pet-test-\(UUID().uuidString).sock"
        var received: [PetEvent] = []
        let server = try SocketServer(path: path) { received.append($0) }
        defer { server.stop() }
        let client = try makeClient(path: path)
        defer { close(client) }
        let speech = String(repeating: "Unicode speech 🐱. ", count: 1000)
        let payload = try JSONSerialization.data(withJSONObject: ["text": speech])
        #expect(payload.count > 16384)
        try send(payload, to: client)
        try await waitUntil { !received.isEmpty }
        try #require(received.count == 1)
        #expect(received[0].text == speech)
    }

    @Test
    func finitePlaybackHoldsLastFrameAndCompletesOnce() throws {
        let command = try #require(event("{\"animation\":{\"id\":\"jump\",\"repeats\":3,\"command_id\":7}}").animation)
        var playback = AnimationPlayback(commandID: command.commandID, clipID: command.id, count: 5, fps: 10,
                                         repeats: command.repeats, start: 100)
        #expect(playback.commandID == 7)
        #expect(playback.sample(at: 100.6).frame == 0)
        #expect(playback.sample(at: 101.4).frame == 4)
        let finished = playback.sample(at: 101.5)
        #expect(finished.frame == 4 && finished.completed)
        #expect(!playback.sample(at: 105).completed)
    }

    @Test
    func repeatingPlaybackContinuesIndefinitely() {
        var loop = AnimationPlayback(commandID: 8, clipID: "rest", count: 2, fps: 2, repeats: nil, start: 0)
        #expect(!loop.sample(at: 100).completed)
    }

    @Test
    func motionStaysWithinMonitorBounds() throws {
        let command = try #require(event("{\"motion\":{\"x\":2000,\"y\":-500,\"seconds\":2,\"command_id\":9}}").motion)
        let size = CGSize(width: 200, height: 160)
        let bounds = CGRect(x: -100, y: 20, width: 1000, height: 700)
        let motion = WindowMotion(command: command, origin: CGPoint(x: 100, y: 100), size: size, bounds: bounds, started: 10)
        let middle = motion.sample(at: 11, size: size, bounds: bounds)
        #expect(middle.origin == CGPoint(x: 400, y: 60) && !middle.finished)
        let final = motion.sample(at: 12, size: size, bounds: bounds)
        #expect(final.origin == CGPoint(x: 700, y: 20) && final.finished)
        let smaller = motion.sample(at: 12, size: size, bounds: CGRect(x: 0, y: 0, width: 100, height: 100))
        #expect(smaller.origin == .zero)
    }

    @Test
    func loadsValidSpritePack() throws {
        let directory = try makeSpritePack(manifest: validManifest)
        defer { try? FileManager.default.removeItem(at: directory) }
        let pack = try SpriteManifest.load(directory: directory)
        #expect(pack.pet == "quiet")
        #expect(pack.rest == "rest")
        #expect(pack.clips.count == 1)
        #expect(pack.clips["rest"]?.frames == ["rest.png"])
    }

    @Test(arguments: [
        validManifest.replacingOccurrences(of: "rest.png", with: "missing.png"),
        validManifest.replacingOccurrences(of: "rest.png", with: "../rest.png"),
        validManifest.replacingOccurrences(of: #""fps":1"#, with: #""fps":0"#),
        validManifest.replacingOccurrences(of: #""rest":"rest""#, with: #""rest":"missing""#)
    ])
    func rejectsInvalidSpritePacks(manifest: String) throws {
        let directory = try makeSpritePack(manifest: manifest)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(throws: (any Error).self) { try SpriteManifest.load(directory: directory) }
    }

    private func makeSpritePack(manifest: String) throws -> URL {
        let directory = URL(fileURLWithPath: "/tmp/pet-pack-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data([0]).write(to: directory.appendingPathComponent("rest.png"))
        try Data(manifest.utf8).write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    private func waitUntil(_ condition: () -> Bool) async throws {
        let deadline = DispatchTime.now() + .seconds(2)
        while !condition() {
            try #require(DispatchTime.now() < deadline, "Timed out waiting for speech expiration or socket delivery")
            try await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    @Test
    func reclaimsStaleSocketAndStopsIdempotently() throws {
        let path = "/tmp/pet-test-\(UUID().uuidString).sock"
        var address = makeAddress(path)
        let stale = socket(AF_UNIX, SOCK_DGRAM, 0)
        try #require(stale >= 0)
        let result = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(stale, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        close(stale)
        try #require(result == 0)
        let server = try SocketServer(path: path) { _ in }
        server.stop()
        server.stop()
        var removed = stat()
        #expect(lstat(path, &removed) == -1 && errno == ENOENT)
    }

    @Test
    func socketStartupPreservesRegularFiles() throws {
        let file = "/tmp/pet-test-\(UUID().uuidString).txt"
        try Data("Keep me".utf8).write(to: URL(fileURLWithPath: file))
        defer { try? FileManager.default.removeItem(atPath: file) }
        #expect(throws: (any Error).self) { try SocketServer(path: file) { _ in } }
        #expect(try String(contentsOfFile: file, encoding: .utf8) == "Keep me")
    }

    private func makeAddress(_ path: String) -> sockaddr_un {
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        withUnsafeMutableBytes(of: &address.sun_path) { $0.copyBytes(from: Array(path.utf8) + [0]) }
        return address
    }

    private func makeClient(path: String) throws -> Int32 {
        let fd = socket(AF_UNIX, SOCK_DGRAM, 0)
        var sendBuffer: Int32 = 131072
        guard setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sendBuffer, socklen_t(MemoryLayout<Int32>.size)) == 0 else {
            close(fd)
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        var address = makeAddress(path)
        let result = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard result == 0 else { close(fd); throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
        return fd
    }

    private func send(_ data: Data, to fd: Int32) throws {
        let sent = data.withUnsafeBytes { Darwin.send(fd, $0.baseAddress, $0.count, 0) }
        guard sent == data.count else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
    }
}

private let validManifest = #"{"pet":"quiet","rest":"rest","clips":{"rest":{"frames":["rest.png"],"fps":1,"role":"rest"}}}"#
