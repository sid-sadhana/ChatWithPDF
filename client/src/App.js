import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [pdfFile, setPdfFile] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (input.trim() && pdfFile) {
      const userMessage = { text: input, sender: 'user' };
      setMessages(prev => [...prev, userMessage]);
      setInput('');

      const formData = new FormData();
      formData.append('query', input);
      formData.append('file', pdfFile, `${uuidv4()}.pdf`);

      try {
        const response = await axios.post('http://localhost:5000/api/generate', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        const botMessage = { text: response.data, sender: 'bot' };
        setMessages(prev => [...prev, botMessage]);
      } catch (error) {
        console.error(error);
      }
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center p-4">
      <h1 className="text-4xl sm:text-5xl font-extrabold mb-6">ChatWithPDF</h1>
      <div className="w-full max-w-3xl bg-gray-800 rounded-2xl shadow-lg flex flex-col flex-1">
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`max-w-2xl break-words p-3 rounded-xl ${
                msg.sender === 'user'
                  ? 'self-end bg-blue-600'
                  : 'self-start bg-gray-700'
              }`}
            >
              {msg.text}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div className="p-4 grid grid-cols-[auto_1fr_auto] gap-3 items-center">
          <label className="cursor-pointer bg-green-500 hover:bg-green-600 transition px-4 py-2 rounded-md text-sm sm:text-base">
            {pdfFile ? 'PDF Selected' : 'Upload PDF'}
            <input
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={e => setPdfFile(e.target.files[0])}
            />
          </label>

          <input
            type="text"
            className="bg-gray-700 rounded-md h-12 px-4 focus:outline-none placeholder-gray-400 w-full"
            placeholder="Type your message..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && sendMessage()}
          />

          <button
            className="bg-blue-600 hover:bg-blue-700 transition px-6 py-2 rounded-md h-12 text-sm sm:text-base"
            onClick={sendMessage}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default App;
