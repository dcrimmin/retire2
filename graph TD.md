```mermaid
graph TD
    subgraph "Interface"
        A["User Interaction (Dash App)"] --> B{"Load Run?"};
        B -- "Yes" --> C["Load Run Data from DB"];
        B -- "No" --> D["Input Parameters (Portfolio, Horizon, etc.)"];
        D --> E["Input Income/Expense Streams"];
        E --> F["Run Simulation Button Clicked"];
        F --> G{"Simulation Running?"};
        G -- "Yes" --> H["Display Loading Spinner"];
        G -- "No" --> I["Display Results/Error"];
        C --> I;
        I --> J{"Display Results?"};
        J -- "Yes" --> K["Display Summary Stats"];
        J -- "No" --> L["Display Error Message"];
        K --> M["Display Histogram"];
        K --> N["Display Key Path Evolution Plot"];
        N --> O["Display Detail Table"];
        L --> A;
    end

    subgraph "Simulation Core"
        S["Run Simulation"] --> T["Generate Stochastic Returns/Inflation"];
        T --> U{"Solve for Initial Withdrawal (Brentq)"};
        U -- "Success" --> V["Run Single Path with Solved W"];
        U -- "Failure" --> W["Record Solver Failure"];
        V --> X["Calculate Yearly Data"];
        X --> Y["Store Path Results"];
        W --> Y;
        Y --> Z{"All Paths Completed?"};
        Z -- "Yes" --> AA["Analyze Results (Percentiles, Mean, Median)"];
        Z -- "No" --> S;
        AA --> BB["Prepare Data for Database"];
    end

    subgraph "Database Interaction"
        DB1["Setup Database (if needed)"] --> DB2["Save Simulation Run Parameters"];
        DB2 --> DB3["Save Income/Expense Streams"];
        DB3 --> DB4["Save Path Results"];
        DB4 --> DB5["Save Yearly Data"];
        BB --> DB2;
    end

    subgraph "Data Flow"
        A -- "Parameters" --> S;
        S -- "Results" --> AA;
        AA -- "Data" --> DB2;
        C -- "Data" --> A;
        DB5 -- "Data" --> N;
        DB5 -- "Data" --> O;
    end

    
    ```