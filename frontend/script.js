async function sendMessage() {
  const input = document.getElementById("message");
  const text = input.value.trim();
  
  if (!text) return;

  const chat = document.getElementById("chat");
  
  // Remove welcome screen if it exists
  const welcomeScreen = chat.querySelector(".welcome-screen");
  if (welcomeScreen) {
    welcomeScreen.remove();
  }

  // Add user message
  const userDiv = document.createElement("div");
  userDiv.className = "message user";
  userDiv.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
  chat.appendChild(userDiv);
  
  input.value = "";
  input.focus();
  scrollToBottom();

  // Disable send button and show loading state
  const sendBtn = document.querySelector(".send-button");
  sendBtn.disabled = true;

  try {
    const res = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();
    
    // Add bot message
    const botDiv = document.createElement("div");
    botDiv.className = "message bot";
    const botContent = document.createElement("div");
    botContent.className = "message-content";
    botContent.innerHTML = escapeHtml(data.reply);
    botDiv.appendChild(botContent);
    chat.appendChild(botDiv);

    // Mark as complete (remove blinking dot)
    setTimeout(() => {
      botContent.classList.add("complete");
    }, 300);
    
    scrollToBottom();
  } catch (error) {
    const errorDiv = document.createElement("div");
    errorDiv.className = "message bot";
    errorDiv.innerHTML = `<div class="message-content complete" style="background: #fee; color: #c33;">Error: ${escapeHtml(error.message)}</div>`;
    chat.appendChild(errorDiv);
    console.error("Error:", error);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

function scrollToBottom() {
  const chat = document.getElementById("chat");
  setTimeout(() => {
    chat.scrollTop = chat.scrollHeight;
  }, 0);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Auto-focus input on page load
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("message").focus();
});