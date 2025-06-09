function sendMessage() {
    const userInput = document.getElementById("user-input");
    const message = userInput.value.trim();

    if (message !== "") {
        addMessage(message, "user");

        fetch("/chat", {
            method: "POST",
            body: JSON.stringify({ message: message }),
            headers: {
                "Content-Type": "application/json"
            }
        })
        .then(response => response.json())
        .then(data => {
            addMessage(data.response, "bot");
        });

        userInput.value = "";
    }
}

function addMessage(message, sender) {
    const chatBox = document.getElementById("chat-box");
    const messageElement = document.createElement("p");
    messageElement.className = sender === "user" ? "user-message" : "bot-message";
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}