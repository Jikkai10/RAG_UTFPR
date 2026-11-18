async function* parseSSEStream(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split("\n\n");
    buffer = parts.pop();

    for (const part of parts) {
      const lines = part.split("\n");

      let event = "message";
      let data = "";

      for (const line of lines) {
        if (line.startsWith("event: ")) {
          event = line.replace("event: ", "");
        }
        if (line.startsWith("data: ")) {
          data += line.replace("data: ", "");
        }
      }

      yield { event, data };
    }
  }
}

export default parseSSEStream;