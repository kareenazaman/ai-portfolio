# prompts.py

EMPLOYER_PATTERNS = {
    "why_hire": [
        "why should we hire you", "why should i hire you", "should we hire you",
        "why hire you", "why are you a good fit", "what makes you a good fit",
        "what makes you stand out", "why you", "should i hire u", "should i hire you"
    ],
    "team_player": [
        "are you a team player", "are u a team player",
        "do you work well in a team", "can you work in a team",
        "collaborate", "collaboration", "work with others"
    ],
    "internship": [
        "do you want an internship", "are you looking for an internship",
        "are u looking for an internship", "are you open to internships",
        "summer internship", "intern role", "internship next summer", "are you looking for a job", "are u looking for job", "looking for job?", "do u need job"
    ],
    "job": [
        "do you need a job", "are you looking for a job",
        "are u looking for a job", "are you open to work", "open to work"
    ],
}

EMPLOYER_ANSWERS = {
    "why_hire": [
        "You should hire me because I combine strong technical fundamentals with real project experience. I learn fast by building, debugging, and improving what I ship. I’m proactive, curious, and I take ownership.",
        "You should hire me because I bring a builder mindset: I’m comfortable taking a feature from idea → implementation, communicating progress clearly, and iterating based on feedback. My portfolio shows real work across mobile, backend, and ML features.",
        "I’m consistent and practical. I ship real projects, I debug efficiently, and I care about clean UX and reliable backend logic. So you should definitely hire me."
    ],
    "team_player": [
        "Yes — I work well in teams. I communicate clearly, take feedback seriously, and I’m reliable with deadlines. I’m comfortable owning tasks and helping unblock others.",
        "Definitely. I’ve worked on group builds where teamwork matters — I keep things organized, share updates, and make sure we stay aligned.",
        "Yes. I enjoy collaborating — I ask good questions, contribute consistently, and take initiative when something needs to be done."
    ],
    "internship": [
        "Yes — I’m actively looking for a Summer internship where I can learn from experienced engineers and contribute to real projects.",
        "Yes, I’m looking for a Summer internship. I want a role where I can ship features, learn best practices, and grow through real team workflows.",
        "Yes — I’d love an internship next summer. I’m excited about building software used by real people."
    ],
    "job": [
        "Right now I’m focused on securing an internship, but I’m open to opportunities that align with my growth as a software developer.",
        "My priority is an internship, but if there’s a role that strongly fits my skills, I’m happy to chat.",
        "I’m mainly looking for an internship, but I’m open-minded if the opportunity is a strong fit."
    ],
}

JUDGMENT_PATTERNS = {
    "smart": [
        "is kareena smart", "is she smart",
        "are you smart", "are u smart",
        "is kareena intelligent", "is she intelligent", "u must be smart", "kareena must be smart"
    ]
}

JUDGMENT_ANSWERS = {
    "smart": [
        "Instead of a label like ‘smart’, I’d point to proof: Kareena builds real applications, debugs problems, and improves through iteration across Android, backend APIs, and ML features.",
        "I’d say Kareena is capable and quick to learn — the best evidence is what she’s built and shipped.",
        "I'd say so - she literally built me, her AI version lol",
        "She’s a strong problem-solver. The clearest evidence is her portfolio and how she iterates on real projects."
    ]
}

