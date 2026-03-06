document.addEventListener("DOMContentLoaded", () => {
    // DOM Element References
    const form = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const sendButton = document.getElementById("send-btn");
    const recordBtn = document.getElementById('record-btn');
    const scrollUpBtn = document.getElementById('scroll-up');
    const scrollDownBtn = document.getElementById('scroll-down');
    const recordingIndicator = document.getElementById('recording-indicator');

    // Status indicators
    const ollamaStatus = document.getElementById('ollama-status');
    const vectordbStatus = document.getElementById('vectordb-status');
    const audioStatus = document.getElementById('audio-status');

    let mediaRecorder = null;
    let audioChunks = [];

    // --- System Status Check ---
    const checkSystemStatus = async () => {
        try {
            const response = await fetch('/health');
            if (response.ok) {
                const status = await response.json();
                
                // Update Ollama status
                if (status.llm_status === 'available') {
                    ollamaStatus.textContent = '● Online';
                    ollamaStatus.className = 'status-indicator online';
                } else {
                    ollamaStatus.textContent = '● Offline';
                    ollamaStatus.className = 'status-indicator offline';
                }
                
                // Update Vector DB status
                if (status.retrieval_status === 'ready') {
                    vectordbStatus.textContent = '● Ready';
                    vectordbStatus.className = 'status-indicator online';
                } else {
                    vectordbStatus.textContent = '● Not Ready';
                    vectordbStatus.className = 'status-indicator offline';
                }
            } else {
                throw new Error('Health check failed');
            }
        } catch (error) {
            console.error('Status check failed:', error);
            ollamaStatus.textContent = '● Offline';
            ollamaStatus.className = 'status-indicator offline';
            vectordbStatus.textContent = '● Error';
            vectordbStatus.className = 'status-indicator offline';
        }
    };

    // Check system status on load
    checkSystemStatus();

    // --- Utility Functions ---
    // ✨ Fix: Added function to detect Arabic characters
    const isArabic = (text) => /[\u0600-\u06FF]/.test(text);
    
    const setUiLoadingState = (isLoading) => {
        userInput.disabled = isLoading;
        sendButton.disabled = isLoading;
        recordBtn.disabled = isLoading;
    };
    
    const escapeHTML = (str) => String(str).replace(/[&<>'"]/g, tag => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'}[tag] || tag));
    const renderMessageHTML = (text) => `<p>${escapeHTML(text).replace(/\n/g, '<br/>')}</p>`;
    const scrollToBottom = () => { chatBox.scrollTop = chatBox.scrollHeight; };

    // --- Core Chat Functions ---
    const appendMessage = (text, sender, isVoice = false) => {
        const messageWrapper = document.createElement("div");
        messageWrapper.classList.add("chat-message", sender);
        const messageContent = document.createElement("div");
        messageContent.classList.add("message-content");
        messageContent.innerHTML = isVoice ? `<p><i>Sent a voice message...</i></p>` : renderMessageHTML(text);

        // ✨ Fix: Add this condition to set text direction for user messages
        if (!isVoice && isArabic(text)) {
            messageContent.dir = 'rtl';
        }

        messageWrapper.appendChild(messageContent);
        chatBox.appendChild(messageWrapper);
        scrollToBottom();
    };

    const createBotMessagePlaceholder = () => {
        const wrapper = document.createElement("div");
        wrapper.classList.add("chat-message", "bot");
        const content = document.createElement("div");
        content.classList.add("message-content");
        content.innerHTML = `<div class="loading-dots"><span></span><span></span><span></span></div>`;
        wrapper.appendChild(content);
        chatBox.appendChild(wrapper);
        scrollToBottom();
        return content;
    };

    // --- Audio Recording Logic ---
    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
            
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                if (audioBlob.size > 100) {
                    sendAudioBlob(audioBlob);
                }
                stream.getTracks().forEach(track => track.stop());
            };
            
            mediaRecorder.start();
            
            // Toggle UI to show indicator
            userInput.style.display = 'none';
            recordingIndicator.style.display = 'flex';
            recordBtn.classList.add('recording');
            recordBtn.title = 'Stop recording';
            recordBtn.innerHTML = '<i class="fas fa-stop"></i>';

        } catch (err) {
            console.error('Microphone access denied or not available:', err);
            alert('Microphone access is required for voice messages.');
            audioStatus.textContent = '● Unavailable';
            audioStatus.className = 'status-indicator offline';
        }
    };

    const stopRecording = () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            
            // Toggle UI back to normal
            userInput.style.display = 'block';
            recordingIndicator.style.display = 'none';
            recordBtn.classList.remove('recording');
            recordBtn.title = 'Record voice message';
            recordBtn.innerHTML = '<i class="fas fa-microphone"></i>';
        }
    };

    // Main record button listener
    recordBtn.addEventListener('click', () => {
        if (!mediaRecorder || mediaRecorder.state === 'inactive') {
            startRecording();
        } else {
            stopRecording();
        }
    });

    // --- API Communication ---
    const sendTextMessage = async (message) => {
        appendMessage(message, "user");
        userInput.value = '';
        userInput.style.height = 'auto';
        setUiLoadingState(true);
        const botMessageElement = createBotMessagePlaceholder();

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            if (!response.body) {
                throw new Error('No response body received');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let streamingText = "";
            let langChecked = false; // ✨ Fix: Add a flag to check language only once

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                streamingText += chunk;
                
                // ✨ Fix: Check on the first non-empty chunk and set direction
                if (!langChecked && streamingText.trim()) {
                    if (isArabic(streamingText)) {
                        botMessageElement.dir = 'rtl';
                    }
                    langChecked = true;
                }

                botMessageElement.innerHTML = renderMessageHTML(streamingText);
                scrollToBottom();
            }

        } catch (error) {
            console.error("Fetch Error:", error);
            botMessageElement.innerHTML = renderMessageHTML("Sorry, I couldn't connect to the server. Please check that Ollama is running and try again.");
        } finally {
            setUiLoadingState(false);
            userInput.focus();
        }
    };
    
    const sendAudioBlob = async (blob) => {
        appendMessage('', 'user', true);
        setUiLoadingState(true);
        const botMessageElement = createBotMessagePlaceholder();

        try {
            const formData = new FormData();
            formData.append('file', blob, 'recording.webm');
            const response = await fetch('/audio-chat', { method: 'POST', body: formData });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Server error during audio processing.' }));
                throw new Error(errorData.error || 'Server error during audio processing.');
            }
            
            const data = await response.json();
            // ✨ Fix: Also check for RTL in audio response text
            if(isArabic(data.text || '')) {
                botMessageElement.dir = 'rtl';
            }
            botMessageElement.innerHTML = renderMessageHTML(data.text || 'Received an empty response.');

            if (data.audio_url) {
                const audio = document.createElement('audio');
                audio.controls = true;
                audio.src = data.audio_url;
                audio.playbackRate = 1.3; // Natural playback speed
                audio.defaultPlaybackRate = 1.3; // Ensure default speed persists
                botMessageElement.appendChild(audio);
                
                const tryPlay = () => {
                    audio.play().catch(e => {
                        console.warn('Audio autoplay was prevented by browser.', e);
                    });
                };
                if (audio.readyState >= 2) {
                    tryPlay();
                } else {
                    audio.addEventListener('canplay', tryPlay, { once: true });
                }
            }

        } catch (err) {
            console.error('Audio Chat Error:', err);
            botMessageElement.innerHTML = renderMessageHTML(`Sorry, I couldn't process your audio: ${err.message}`);
        } finally {
            setUiLoadingState(false);
            userInput.focus();
        }
    };

    // --- Event Listeners ---
    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const message = userInput.value.trim();
        if (message) {
            sendTextMessage(message);
        }
    });
    
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        const maxHeight = 160;
        userInput.style.height = `${Math.min(userInput.scrollHeight, maxHeight)}px`;
    });

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
        }
    });

    // --- Scroll Controls ---
    const updateScrollButtons = () => {
        const isAtBottom = Math.abs(chatBox.scrollHeight - chatBox.clientHeight - chatBox.scrollTop) < 5;
        const isAtTop = chatBox.scrollTop < 5;
        scrollDownBtn.classList.toggle('visible', !isAtBottom);
        scrollUpBtn.classList.toggle('visible', !isAtTop && chatBox.scrollHeight > chatBox.clientHeight);
    };

    chatBox.addEventListener('scroll', updateScrollButtons, { passive: true });
    window.addEventListener('resize', updateScrollButtons);
    scrollDownBtn.addEventListener('click', () => { 
        chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: 'smooth' }); 
    });
    scrollUpBtn.addEventListener('click', () => { 
        chatBox.scrollBy({ top: -chatBox.clientHeight * 0.8, behavior: 'smooth' }); 
    });
    
    // Initialize scroll buttons
    setTimeout(updateScrollButtons, 100);

    // Focus on input when page loads
    userInput.focus();
});