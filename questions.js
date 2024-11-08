const sections = [
    {
      title: 'GENERAL APPEARANCE',
      questions: [
        {
          question: "Imagine you have an important client meeting scheduled for tomorrow. How would you choose your attire for the meeting, and why do you think it's important to dress appropriately in this situation?",
          criteria: [
            { label: "Understanding of dress code" },
            { label: "Appropriateness of attire" },
            { label: "Importance of impression" },
          ],
        },
        {
          question: "You have been assigned to represent the company at a networking event. Think of your grooming routine before attending the event. Why do you think personal grooming is crucial in this scenario?",
          criteria: [
            { label: "Completeness of routine" },
            { label: "Attention to detail" },
            { label: "Importance of grooming" },
          ],
        },
        {
          question: "You have a long day ahead with several back-to-back meetings. How do you maintain your personal hygiene throughout the day, and why is it important for your role?",
          criteria: [
            { label: "Practicality of strategies" },
            { label: "Consistency in hygiene" },
            { label: "Importance of hygiene" },
          ],
        },
        {
          question: "During a presentation to senior management, how would you ensure your posture and poise convey confidence and professionalism? Why is this important?",
          criteria: [
            { label: "Techniques for posture" },
            { label: "Confident body language" },
            { label: "Impact of posture and poise" },
          ],
        },
        {
          question: "You are preparing for a job interview panel. How would you choose your accessories and jewelry, and what considerations would you keep in mind to ensure they are appropriate?",
          criteria: [
            { label: "Appropriateness of accessories" },
            { label: "Balance of style and professionalism" },
            { label: "Consideration of norms" },
          ],
        },
      ],
    },
    {
      title: 'MANNER OF SPEAKING',
      questions: [
        {
          question: "Imagine you are giving a presentation to a group of colleagues who are unfamiliar with the technical details of the project you're working on. How would you ensure that your speech is clear and your points are well-articulated so that everyone understands?",
          criteria: [
            { label: "Clarity of speech" },
            { label: "Pronunciation of words" },
            { label: "Avoidance of mumbling/slurring" },
          ],
        },
        {
          question: "During a team meeting, you need to address a sensitive issue about project delays. How would you modulate your tone to convey the seriousness of the situation while keeping the team's morale high?",
          criteria: [
            { label: "Use of appropriate tone" },
            { label: "Effective modulation" },
            { label: "Avoidance of monotone" },
          ],
        },
        {
          question: "You are in a client meeting, and they are questioning the feasibility of your proposed solution. How would you confidently and assertively defend your proposal while maintaining a positive and professional demeanor?",
          criteria: [
            { label: "Display of confidence" },
            { label: "Assertiveness" },
            { label: "Positive body language" },
          ],
        },
        {
          question: "While interacting with a diverse team from different cultural backgrounds, how would you ensure your language and communication style are appropriate and respectful to everyone?",
          criteria: [
            { label: "Appropriateness of language" },
            { label: "Respectfulness" },
            { label: "Sensitivity to nuances" },
          ],
        },
        {
          question: "In a brainstorming session, a colleague presents an idea that is different from yours. How would you actively listen and respond to their suggestion in a way that shows you value their input?",
          criteria: [
            { label: "Demonstrates active listening" },
            { label: "Provides relevant responses" },
            { label: "Asks clarifying questions" },
          ],
        },
      ],
    },
    {
      title: 'PHYSICAL CONDITION',
      questions: [
        {
          question: "Imagine you have to work extended hours to meet a critical project deadline. How would you manage your physical stamina and endurance to ensure you remain productive and focused?",
          criteria: [
            { label: "Awareness of limits" },
            { label: "Strategies for energy" },
            { label: "Ability to stay focused" },
          ],
        },
        {
          question: "Your role occasionally requires moving heavy equipment or materials. How do you ensure you can handle these physical tasks safely and efficiently?",
          criteria: [
            { label: "Techniques for lifting" },
            { label: "Awareness of safe practices" },
            { label: "Efficiency in physical tasks" },
          ],
        },
        {
          question: "How do you maintain your overall health and wellness to ensure you can perform your job duties effectively? Think of your routine for staying healthy.",
          criteria: [
            { label: "Consistency in routines" },
            { label: "Importance of health" },
            { label: "Balance of work and health" },
          ],
        },
        {
          question: "During peak periods, you may experience high levels of stress. How do you manage stress to ensure it does not impact your physical condition and job performance?",
          criteria: [
            { label: "Strategies for managing stress" },
            { label: "Effectiveness of techniques" },
            { label: "Awareness of stress impact" },
          ],
        },
        {
          question: "Your job may require you to adapt to varying physical demands, such as different work environments or tasks. How do you ensure you can adapt to these physical demands?",
          criteria: [
            { label: "Flexibility in tasks" },
            { label: "Quick adjustment" },
            { label: "Awareness of adaptability" },
          ],
        },
      ],
    },
    {
      title: 'MENTAL ALERTNESS',
      questions: [
        {
          question: "Imagine you are working on a critical report that requires precision and accuracy. How do you ensure that you pay attention to every detail and avoid mistakes?",
          criteria: [
            { label: "Techniques for focus" },
            { label: "Strategies for double-checking" },
            { label: "Importance of Accuracy" },
          ],
        },
        {
          question: "You encounter an unexpected issue in a project that threatens to delay the deadline. Carefully think about your process for quickly identifying and solving the problem.",
          criteria: [
            { label: "Speed of identification" },
            { label: "Effectiveness of solution" },
            { label: "Calmness under pressure" },
          ],
        },
        {
          question: "During a team meeting, you are asked to make a quick decision on a pressing matter. How do you approach making this decision, and what factors do you consider?",
          criteria: [
            { label: "Speed of decision-making" },
            { label: "Consideration of factors" },
            { label: "Confidence in decision" },
          ],
        },
        {
          question: "Halfway through a project, you receive new information that requires you to change your approach. How do you adapt to this new information and ensure the project stays on track?",
          criteria: [
            { label: "Flexibility in plans" },
            { label: "Speed of adaptation" },
            { label: "Effectiveness of new approach" },
          ],
        },
        {
          question: "You are assigned to handle multiple tasks simultaneously with tight deadlines. How do you manage your time and prioritize tasks to ensure all are completed efficiently?",
          criteria: [
            { label: "Prioritization ability" },
            { label: "Time management skills" },
            { label: "Task efficiency" },
          ],
        },
      ],
    },
    {
    title: 'SELF CONFIDENCE',
    questions: [
      {
        question: "You are asked to present a new project idea to senior management. How do you prepare for the presentation to ensure you come across as confident and knowledgeable?",
        criteria: [
          { label: "Clarity of presentation" },
          { label: "Confidence in delivery" },
          { label: "Engagement with audience" },
        ],
      },
      {
        question: "During a team meeting, you are required to make a quick decision on an important matter. Imagine your process for making a confident decision.",
        criteria: [
          { label: "Speed of decision-making" },
          { label: "Confidence in decision" },
          { label: "Justification" },
        ],
      },
      {
        question: "You receive constructive criticism from your supervisor about your recent work. How do you handle this feedback and demonstrate self-confidence in your response?",
        criteria: [
          { label: "Openness to feedback" },
          { label: "Confidence in response" },
          { label: "Action plan" },
        ],
      },
      {
        question: "You are involved in a disagreement with a colleague. How do you approach resolving the conflict while maintaining your self-confidence?",
        criteria: [
          { label: "Calmness in conflict" },
          { label: "Confidence in expressing views" },
          { label: "Effectiveness in resolution" },
        ],
      },
      {
        question: "You notice an area in your work that could be improved, but it is not part of your usual responsibilities. How do you take the initiative to address this issue confidently?",
        criteria: [
          { label: "Proactiveness" },
          { label: "Confidence in action" },
          { label: "Communication of initiative" },
        ],
      },
    ],
  },
  {
    title: 'ABILITY TO PRESENT IDEAS',
    questions: [
      {
        question: "Can you think of a complex project you worked on, and imagine explaining it in a way that someone not familiar with the field could understand?",
        criteria: [
          { label: "Simplicity of explanation" },
          { label: "Logical structure" },
          { label: "Use of analogies/examples" },
        ],
      },
      {
        question: "How do you ensure your audience stays engaged during your presentations? Provide an example of a technique you use.",
        criteria: [
          { label: "Effectiveness of techniques" },
          { label: "Adaptability to feedback" },
          { label: "Use of interactive elements" },
        ],
      },
      {
        question: "Think of how you utilize visual aids and tools in your presentations. Why do you think they are important?",
        criteria: [
          { label: "Appropriateness of aids" },
          { label: "Clarity and design" },
          { label: "Integration with content" },
        ],
      },
      {
        question: "How do you handle questions and feedback during or after your presentations? Share an example of a challenging question you managed effectively.",
        criteria: [
          { label: "Poise and confidence" },
          { label: "Clarity of responses" },
          { label: "Constructive feedback handling" },
        ],
      },
      {
        question: "Give an example of a time when your presentation successfully persuaded your audience. What techniques did you use to ensure your message was impactful?",
        criteria: [
          { label: "Use of persuasive techniques" },
          { label: "Emotional appeal" },
          { label: "Evidence and logic" },
        ],
      },
    ],
  },
  {
    title: 'COMMUNICATION SKILLS',
    questions: [
      {
        question: "Imagine a situation where you had to explain a complex idea to a colleague or client. How did you ensure that your explanation was clear and concise?",
        criteria: [
          { label: "Clarity" },
          { label: "Conciseness" },
          { label: "Avoidance of jargon" },
        ],
      },
      {
        question: "Can you give an example of a time when active listening helped you solve a problem or improve a situation? What techniques did you use to ensure you were effectively listening?",
        criteria: [
          { label: "Active listening" },
          { label: "Clarifying questions" },
          { label: "Paraphrasing and summarizing" },
        ],
      },
      {
        question: "Imagine in a situation how you use nonverbal communication (such as body language, facial expressions, and eye contact) to enhance your message during a presentation or meeting.",
        criteria: [
          { label: "Consistency with verbal message" },
          { label: "Effectiveness of body language" },
          { label: "Engagement through eye contact" },
        ],
      },
      {
        question: "How do you adjust your communication style when dealing with different audiences (e.g., colleagues, clients, senior management)? Provide an example.",
        criteria: [
          { label: "Understanding of audience" },
          { label: "Flexibility in style" },
          { label: "Effectiveness of adaptation" },
        ],
      },
      {
        question: "Think of a time when you successfully persuaded someone to adopt your point of view or take a particular action. What communication strategies did you use?",
        criteria: [
          { label: "Use of persuasive techniques" },
          { label: "Strength of argument" },
          { label: "Addressing counter arguments" },
        ],
      },
    ],
  },
  {
    title: 'STUDENT PERFORMANCE RATING',
    questions: [
      {
        question: "Visualize a recent project or assignment you worked on. What was the topic, and how did you approach completing it?",
        criteria: [
          { label: "Depth of understanding" },
          { label: "Ability to apply knowledge" },
          { label: "Creativity in approach" },
        ],
      },
      {
        question: "How do you contribute to class discussions and activities? Can you give an example of a time when your participation had a positive impact?",
        criteria: [
          { label: "Frequency of participation" },
          { label: "Quality of contributions" },
          { label: "Impact on class dynamics" },
        ],
      },
      {
        question: "Can you imagine a group project you were involved in? How did you contribute to the team's success, and what role did you play?",
        criteria: [
          { label: "Ability to work with others" },
          { label: "Contribution to team goals" },
          { label: "Leadership and support roles" },
        ],
      },
      {
        question: "How do you manage your time and stay organized with multiple assignments and deadlines? Share a specific strategy that works for you.",
        criteria: [
          { label: "Effectiveness of time management" },
          { label: "Organizational skills" },
          { label: "Ability to meet deadlines" },
        ],
      },
      {
        question: "Envision a challenging problem you faced in your studies and how you solved it. What steps did you take, and what was the outcome?",
        criteria: [
          { label: "Analytical thinking" },
          { label: "Creativity in problem-solving" },
          { label: "Persistence and resourcefulness" },
        ],
      },
    ],
  },
  ];


  function questions() {
    const container = document.getElementById('questionnaire');
    if (!container) {
      console.error("Element with id 'questionnaire' not found.");
      return;
    }
  
    sections.forEach((section) => {
      const sectionTitle = document.createElement('h2');
      sectionTitle.innerText = section.title;
      container.appendChild(sectionTitle);
  
      section.questions.forEach((questionObj) => {
        const questionEl = document.createElement('p');
        questionEl.innerText = questionObj.question;
        container.appendChild(questionEl);
  
        const criteriaList = document.createElement('ul');
        questionObj.criteria.forEach((criterion) => {
          const criterionItem = document.createElement('li');
          criterionItem.innerText = criterion.label;
          criteriaList.appendChild(criterionItem);
        });
        container.appendChild(criteriaList);
      });
    });
  }